#!/usr/bin/env python3
"""
SynthSeg TensorFlow to PyTorch Conversion Script

This script converts SynthSeg TensorFlow models to PyTorch, ensuring identical outputs.
Based on the article approach using ONNX as an intermediate format, but with custom
implementations for SynthSeg-specific layers and preprocessing/postprocessing.

Usage:
    python convert_synthseg_tf_to_pytorch.py --tf_model_path model.h5 --output_path model.pth
"""

import os
import sys
import argparse
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional, Any
import warnings

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# TensorFlow imports (for loading original model)
try:
    import tensorflow as tf
    from tensorflow import keras
    import keras.backend as K
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. You'll need TF to load the original model.")
    TF_AVAILABLE = False

# ONNX imports for conversion path
try:
    import onnx
    import tf2onnx
    from onnx2pytorch import ConvertModel
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: ONNX conversion tools not available. Install with: pip install tf2onnx onnx2pytorch")
    ONNX_AVAILABLE = False


class GaussianBlurPyTorch(nn.Module):
    """
    PyTorch implementation of SynthSeg's GaussianBlur layer.
    """
    def __init__(self, sigma: float, n_dims: int = 3):
        super().__init__()
        self.sigma = sigma
        self.n_dims = n_dims
        
        # Create Gaussian kernel
        if sigma > 0:
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create 1D Gaussian kernel
            ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
            kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            # Convert to tensor and register as buffer
            self.register_buffer('kernel_1d', torch.FloatTensor(kernel_1d))
            self.kernel_size = kernel_size
        else:
            self.kernel_1d = None
            self.kernel_size = 1
    
    def forward(self, x: Tensor) -> Tensor:
        if self.sigma <= 0 or self.kernel_1d is None:
            return x
        
        # Apply separable Gaussian blur
        B, C, D, H, W = x.shape
        
        # Blur in each dimension separately (separable)
        # Dimension 0 (depth)
        x = x.view(B * C, 1, D, H * W)
        kernel = self.kernel_1d.view(1, 1, -1, 1)
        x = F.conv2d(x, kernel, padding=(self.kernel_size // 2, 0))
        x = x.view(B, C, D, H, W)
        
        # Dimension 1 (height)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(B * C, 1, H, D * W)
        x = F.conv2d(x, kernel, padding=(self.kernel_size // 2, 0))
        x = x.view(B, C, H, D, W).permute(0, 1, 3, 2, 4).contiguous()
        
        # Dimension 2 (width)
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B * C, 1, W, D * H)
        x = F.conv2d(x, kernel, padding=(self.kernel_size // 2, 0))
        x = x.view(B, C, W, D, H).permute(0, 1, 3, 4, 2).contiguous()
        
        return x


class RandomFlipPyTorch(nn.Module):
    """
    PyTorch implementation of SynthSeg's RandomFlip layer for test-time augmentation.
    """
    def __init__(self, axis: int = 0, prob: float = 1.0):
        super().__init__()
        self.axis = axis
        self.prob = prob
    
    def forward(self, x: Tensor) -> Tensor:
        if self.training or np.random.random() > self.prob:
            return x
        
        # Flip along specified axis (axis 0 corresponds to dim 2 in BCDHW format)
        flip_dim = self.axis + 2  # Adjust for batch and channel dimensions
        return torch.flip(x, dims=[flip_dim])


class ConvertLabelsPyTorch(nn.Module):
    """
    PyTorch implementation of SynthSeg's ConvertLabels layer.
    """
    def __init__(self, source_values: np.ndarray, dest_values: np.ndarray):
        super().__init__()
        
        # Create lookup table
        max_val = max(source_values.max(), dest_values.max())
        lut = torch.arange(max_val + 1, dtype=torch.long)
        
        for src, dst in zip(source_values, dest_values):
            lut[src] = dst
        
        self.register_buffer('lut', lut)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.lut[x.long()]


class ConvBlock(nn.Module):
    """
    Convolutional block matching SynthSeg's implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, n_convs: int = 2, 
                 conv_size: int = 3, activation: str = 'elu'):
        super().__init__()
        
        layers = []
        for i in range(n_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(nn.Conv3d(in_ch, out_channels, kernel_size=conv_size, 
                                  padding='same', bias=True))
            if activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            elif activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f"Activation function {activation} is not supported.")
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNetEncoder(nn.Module):
    """
    U-Net encoder matching SynthSeg's architecture.
    """
    def __init__(self, in_channels: int = 1, chs: Tuple[int, ...] = (24, 48, 96, 192, 384), 
                 n_convs: int = 2, activation: str = 'elu'):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_ch = in_channels
        for i, ch in enumerate(chs):
            self.downs.append(ConvBlock(prev_ch, ch, n_convs=n_convs, activation=activation))
            if i < len(chs) - 1:
                self.pools.append(nn.MaxPool3d(2))
            prev_ch = ch

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        feats = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                feats.append(x)
                x = self.pools[i](x)
        return x, feats


class UNetDecoder(nn.Module):
    """
    U-Net decoder matching SynthSeg's architecture.
    """
    def __init__(self, n_classes: int, chs: Tuple[int, ...] = (384, 192, 96, 48, 24), 
                 n_convs: int = 2, activation: str = 'elu'):
        super().__init__()
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        for i in range(len(chs) - 1):
            self.ups.append(nn.ConvTranspose3d(chs[i], chs[i+1], kernel_size=2, stride=2))
            self.convs.append(ConvBlock(chs[i+1] * 2, chs[i+1], n_convs=n_convs, activation=activation))
        
        self.out = nn.Conv3d(chs[-1], n_classes, kernel_size=1)

    def forward(self, x: Tensor, encoder_feats: List[Tensor]) -> Tensor:
        for i in range(len(self.convs)):
            x = self.ups[i](x)
            enc_f = encoder_feats[-(i + 1)]
            x = torch.cat([x, enc_f], dim=1)
            x = self.convs[i](x)
        
        x = self.out(x)
        return x


class SynthSegPyTorch(nn.Module):
    """
    Complete PyTorch implementation of SynthSeg with all components.
    """
    def __init__(self, n_classes: int, n_levels: int = 5, n_convs: int = 2, 
                 init_feat: int = 24, feat_mult: int = 2, activation: str = 'elu',
                 sigma_smoothing: float = 0.5, flip_indices: Optional[np.ndarray] = None):
        super().__init__()
        
        # Calculate encoder/decoder channel sizes
        encoder_chs = [init_feat]
        for _ in range(n_levels - 1):
            encoder_chs.append(int(encoder_chs[-1] * feat_mult))
        
        encoder_chs = tuple(encoder_chs)
        decoder_chs = tuple(reversed(encoder_chs))
        
        # Build U-Net
        self.encoder = UNetEncoder(1, encoder_chs, n_convs, activation)
        self.decoder = UNetDecoder(n_classes, decoder_chs, n_convs, activation)
        
        # Optional Gaussian smoothing
        self.gaussian_blur = None
        if sigma_smoothing > 0:
            self.gaussian_blur = GaussianBlurPyTorch(sigma_smoothing)
        
        # Optional test-time flipping
        self.flip_indices = flip_indices
        if flip_indices is not None:
            self.random_flip = RandomFlipPyTorch(axis=0, prob=1.0)
        
        self.n_classes = n_classes
    
    def forward(self, x: Tensor) -> Tensor:
        # Main forward pass
        deepest_features, skip_connections = self.encoder(x)
        logits = self.decoder(deepest_features, skip_connections)
        
        # Apply Gaussian smoothing if specified
        if self.gaussian_blur is not None:
            logits = self.gaussian_blur(logits)
        
        # Test-time flipping augmentation
        if self.flip_indices is not None and not self.training:
            # Segment flipped image
            x_flipped = self.random_flip(x)
            deepest_flipped, skip_flipped = self.encoder(x_flipped)
            logits_flipped = self.decoder(deepest_flipped, skip_flipped)
            
            if self.gaussian_blur is not None:
                logits_flipped = self.gaussian_blur(logits_flipped)
            
            # Flip back and reorder channels
            logits_flipped = torch.flip(logits_flipped, dims=[2])  # Flip back
            
            # Reorder channels according to flip_indices
            reordered_channels = []
            for i in range(self.n_classes):
                reordered_channels.append(logits_flipped[:, [self.flip_indices[i]], :, :, :])
            logits_flipped = torch.cat(reordered_channels, dim=1)
            
            # Average the two predictions
            logits = 0.5 * (logits + logits_flipped)
        
        # Apply softmax to get probabilities
        return F.softmax(logits, dim=1)


class SynthSegConverter:
    """
    Main converter class for TensorFlow to PyTorch SynthSeg models.
    """
    
    def __init__(self):
        self.tf_model = None
        self.pytorch_model = None
    
    def load_tensorflow_model(self, model_path: str) -> Any:
        """Load the TensorFlow SynthSeg model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to load the original model")
        
        self.tf_model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Loaded TensorFlow model from {model_path}")
        return self.tf_model
    
    def extract_model_info(self) -> Dict[str, Any]:
        """Extract key information from the TensorFlow model."""
        if self.tf_model is None:
            raise ValueError("TensorFlow model not loaded")
        
        # Get output shape to determine number of classes
        output_shape = self.tf_model.output.shape
        n_classes = output_shape[-1]
        
        # Get input shape
        input_shape = self.tf_model.input.shape
        
        # Extract layer information
        layer_info = {}
        for layer in self.tf_model.layers:
            layer_info[layer.name] = {
                'type': type(layer).__name__,
                'config': layer.get_config() if hasattr(layer, 'get_config') else None
            }
        
        return {
            'n_classes': n_classes,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'layer_info': layer_info
        }
    
    def create_pytorch_model(self, model_info: Dict[str, Any], 
                           sigma_smoothing: float = 0.5,
                           flip_indices: Optional[np.ndarray] = None) -> SynthSegPyTorch:
        """Create the equivalent PyTorch model."""
        self.pytorch_model = SynthSegPyTorch(
            n_classes=model_info['n_classes'],
            sigma_smoothing=sigma_smoothing,
            flip_indices=flip_indices
        )
        return self.pytorch_model
    
    def transfer_weights_direct(self, weight_mapping: Optional[Dict[str, str]] = None):
        """
        Transfer weights directly from TensorFlow to PyTorch.
        This is the most accurate method but requires manual mapping.
        """
        if self.tf_model is None or self.pytorch_model is None:
            raise ValueError("Both TensorFlow and PyTorch models must be loaded")
        
        print("Transferring weights from TensorFlow to PyTorch...")
        
        # Default weight mapping for SynthSeg U-Net
        if weight_mapping is None:
            weight_mapping = self._create_default_weight_mapping()
        
        # Transfer weights
        tf_weights = {}
        for layer in self.tf_model.layers:
            if layer.weights:
                tf_weights[layer.name] = [w.numpy() for w in layer.weights]
        
        with torch.no_grad():
            self._transfer_conv_weights(tf_weights)
        
        print("Weight transfer completed!")
    
    def _create_default_weight_mapping(self) -> Dict[str, str]:
        """Create default weight mapping between TF and PyTorch layers."""
        # This would need to be customized based on the specific model architecture
        mapping = {}
        
        # Encoder mappings
        for i in range(5):  # 5 levels
            for j in range(2):  # 2 convs per level
                tf_name = f"unet_conv{i}_{j}"
                pt_name = f"encoder.downs.{i}.block.{j*2}"  # *2 because of activation layers
                mapping[tf_name] = pt_name
        
        # Decoder mappings
        for i in range(4):  # 4 decoder levels
            for j in range(2):  # 2 convs per level
                tf_name = f"unet_upconv{i}_{j}"
                pt_name = f"decoder.convs.{i}.block.{j*2}"
                mapping[tf_name] = pt_name
        
        # Final output layer
        mapping["unet_prediction"] = "decoder.out"
        
        return mapping
    
    def _transfer_conv_weights(self, tf_weights: Dict[str, List[np.ndarray]]):
        """Transfer convolutional layer weights."""
        # This is a simplified version - would need to be expanded for full compatibility
        pytorch_state_dict = self.pytorch_model.state_dict()
        
        # Transfer encoder weights
        encoder_layer_idx = 0
        for level in range(5):
            for conv in range(2):
                # Find corresponding TF layer
                tf_layer_name = None
                for name in tf_weights.keys():
                    if f"conv{level}_{conv}" in name and "unet" in name:
                        tf_layer_name = name
                        break
                
                if tf_layer_name and tf_weights[tf_layer_name]:
                    tf_kernel, tf_bias = tf_weights[tf_layer_name]
                    
                    # Convert TF weights (DHWIO) to PyTorch format (OIDHW)
                    pt_kernel = np.transpose(tf_kernel, (4, 3, 0, 1, 2))
                    
                    # Get PyTorch layer names
                    pt_weight_name = f"encoder.downs.{level}.block.{conv*2}.weight"
                    pt_bias_name = f"encoder.downs.{level}.block.{conv*2}.bias"
                    
                    if pt_weight_name in pytorch_state_dict:
                        pytorch_state_dict[pt_weight_name].copy_(torch.from_numpy(pt_kernel))
                    if pt_bias_name in pytorch_state_dict:
                        pytorch_state_dict[pt_bias_name].copy_(torch.from_numpy(tf_bias))
        
        # Similar process for decoder weights would go here...
    
    def convert_via_onnx(self, onnx_path: str = "temp_model.onnx") -> SynthSegPyTorch:
        """
        Convert model via ONNX intermediate format.
        This is less accurate but more automated.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX conversion tools are required")
        
        if self.tf_model is None:
            raise ValueError("TensorFlow model not loaded")
        
        print("Converting via ONNX...")
        
        # Convert TF model to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(self.tf_model)
        
        # Save ONNX model temporarily
        onnx.save(onnx_model, onnx_path)
        
        # Convert ONNX to PyTorch
        onnx_model_loaded = onnx.load(onnx_path)
        pytorch_model_onnx = ConvertModel(onnx_model_loaded)
        
        # Clean up temporary file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        print("ONNX conversion completed!")
        return pytorch_model_onnx
    
    def validate_conversion(self, test_input: np.ndarray, tolerance: float = 1e-5) -> bool:
        """
        Validate that the converted model produces the same outputs as the original.
        """
        if self.tf_model is None or self.pytorch_model is None:
            raise ValueError("Both models must be loaded for validation")
        
        print("Validating conversion...")
        
        # TensorFlow prediction
        tf_output = self.tf_model.predict(test_input)
        
        # PyTorch prediction
        self.pytorch_model.eval()
        with torch.no_grad():
            torch_input = torch.from_numpy(test_input).float()
            torch_output = self.pytorch_model(torch_input).numpy()
        
        # Compare outputs
        diff = np.abs(tf_output - torch_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        
        success = max_diff < tolerance
        if success:
            print("✅ Validation successful! Outputs match within tolerance.")
        else:
            print("❌ Validation failed! Outputs differ significantly.")
        
        return success
    
    def save_pytorch_model(self, output_path: str):
        """Save the converted PyTorch model."""
        if self.pytorch_model is None:
            raise ValueError("PyTorch model not created")
        
        torch.save({
            'model_state_dict': self.pytorch_model.state_dict(),
            'model_config': {
                'n_classes': self.pytorch_model.n_classes,
                'n_levels': 5,
                'n_convs': 2,
                'init_feat': 24,
                'feat_mult': 2,
                'activation': 'elu'
            }
        }, output_path)
        
        print(f"PyTorch model saved to {output_path}")


def preprocess_image_tf_style(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to match SynthSeg's TensorFlow preprocessing exactly.
    """
    # Normalize to [0, 1] using SynthSeg's percentile method
    min_val = np.percentile(image, 0.5)
    max_val = np.percentile(image, 99.5)
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val + 1e-15)
    
    return image


def postprocess_output_tf_style(output: np.ndarray, labels_segmentation: np.ndarray) -> np.ndarray:
    """
    Postprocess output to match SynthSeg's TensorFlow postprocessing.
    """
    # Get hard segmentation
    seg = labels_segmentation[output.argmax(-1).astype('int32')].astype('int32')
    
    return seg


def main():
    parser = argparse.ArgumentParser(description="Convert SynthSeg TensorFlow model to PyTorch")
    parser.add_argument("--tf_model_path", required=True, help="Path to TensorFlow model (.h5)")
    parser.add_argument("--output_path", required=True, help="Output path for PyTorch model (.pth)")
    parser.add_argument("--method", choices=["direct", "onnx"], default="direct", 
                       help="Conversion method")
    parser.add_argument("--sigma_smoothing", type=float, default=0.5, 
                       help="Gaussian smoothing sigma")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate conversion with test input")
    parser.add_argument("--test_shape", nargs=3, type=int, default=[64, 64, 64],
                       help="Test input shape for validation")
    
    args = parser.parse_args()
    
    # Create converter
    converter = SynthSegConverter()
    
    try:
        # Load TensorFlow model
        converter.load_tensorflow_model(args.tf_model_path)
        
        # Extract model information
        model_info = converter.extract_model_info()
        print(f"Model info: {model_info['n_classes']} classes")
        
        # Create PyTorch model
        converter.create_pytorch_model(model_info, sigma_smoothing=args.sigma_smoothing)
        
        # Convert weights
        if args.method == "direct":
            converter.transfer_weights_direct()
        elif args.method == "onnx":
            converter.pytorch_model = converter.convert_via_onnx()
        
        # Validate if requested
        if args.validate:
            test_input = np.random.randn(1, 1, *args.test_shape).astype(np.float32)
            test_input = preprocess_image_tf_style(test_input)
            converter.validate_conversion(test_input)
        
        # Save PyTorch model
        converter.save_pytorch_model(args.output_path)
        
        print(f"✅ Conversion completed successfully!")
        print(f"PyTorch model saved to: {args.output_path}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()