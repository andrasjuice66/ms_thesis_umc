# brain_age_pred/utils/synthseg_transfer.py

import torch
import torch.nn as nn
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from collections import defaultdict
import textwrap

class ArchitectureVisualizer:
    """
    Visualize and compare SynthSeg and PyTorch model architectures
    """
    
    def __init__(self, synthseg_weights: Dict, pytorch_model: nn.Module):
        self.synthseg_weights = synthseg_weights
        self.pytorch_model = pytorch_model
        self.pytorch_state_dict = pytorch_model.state_dict()
    
    def print_architecture_comparison(self):
        """Print side-by-side comparison of both architectures"""
        print("\n" + "="*120)
        print("🏗️  ARCHITECTURE COMPARISON: SYNTHSEG ↔️ PYTORCH")
        print("="*120)
        
        self._print_side_by_side_comparison()
        self._print_detailed_layer_mapping()
        self._print_parameter_statistics()
    
    def _print_side_by_side_comparison(self):
        """Print side-by-side architecture overview"""
        print(f"\n{'SYNTHSEG (Keras/TensorFlow)':^58} │ {'PYTORCH (MultiTaskBrainAge)':^58}")
        print("─" * 58 + "┼" + "─" * 58)
        
        # Parse SynthSeg layers
        synthseg_layers = self._parse_synthseg_layers()
        pytorch_layers = self._parse_pytorch_layers()
        
        max_lines = max(len(synthseg_layers), len(pytorch_layers))
        
        for i in range(max_lines):
            left = synthseg_layers[i] if i < len(synthseg_layers) else ""
            right = pytorch_layers[i] if i < len(pytorch_layers) else ""
            print(f"{left:58} │ {right:58}")
    
    def _parse_synthseg_layers(self) -> List[str]:
        """Parse SynthSeg architecture from weights"""
        layers = []
        
        # Input
        layers.append("🔵 INPUT: [B, H, W, D, 1]")
        layers.append("")
        
        # Encoder
        layers.append("🔽 ENCODER (Downsampling Path):")
        encoder_channels = [24, 48, 96, 192, 384]
        
        for level, ch in enumerate(encoder_channels):
            layers.append(f"  Level {level} (→{ch} channels):")
            
            # Check if conv layers exist
            conv0_key = f"unet_conv_downarm_{level}_0/unet_conv_downarm_{level}_0/kernel:0"
            conv1_key = f"unet_conv_downarm_{level}_1/unet_conv_downarm_{level}_1/kernel:0"
            
            if conv0_key in self.synthseg_weights:
                shape = self.synthseg_weights[conv0_key].shape
                in_ch = shape[3]
                layers.append(f"    Conv3D: {in_ch}→{ch}, kernel=3x3x3")
            
            layers.append(f"    BatchNorm3D({ch})")
            layers.append(f"    ELU()")
            
            if conv1_key in self.synthseg_weights:
                layers.append(f"    Conv3D: {ch}→{ch}, kernel=3x3x3")
            
            layers.append(f"    BatchNorm3D({ch})")
            layers.append(f"    ELU()")
            
            if level < len(encoder_channels) - 1:
                layers.append(f"    MaxPool3D(2)")
            layers.append("")
        
        # Decoder
        layers.append("🔼 DECODER (Upsampling Path):")
        decoder_levels = [(5, 192), (6, 96), (7, 48), (8, 24)]
        
        for synthseg_level, ch in decoder_levels:
            layers.append(f"  Level {synthseg_level} (→{ch} channels):")
            layers.append(f"    UpSampling3D(2)")
            layers.append(f"    Concatenate with skip connection")
            
            conv0_key = f"unet_conv_uparm_{synthseg_level}_0/unet_conv_uparm_{synthseg_level}_0/kernel:0"
            if conv0_key in self.synthseg_weights:
                shape = self.synthseg_weights[conv0_key].shape
                in_ch = shape[3]
                layers.append(f"    Conv3D: {in_ch}→{ch}, kernel=3x3x3")
            
            layers.append(f"    BatchNorm3D({ch})")
            layers.append(f"    ELU()")
            layers.append(f"    Conv3D: {ch}→{ch}, kernel=3x3x3")
            layers.append(f"    BatchNorm3D({ch})")
            layers.append(f"    ELU()")
            layers.append("")
        
        # Final layer
        layers.append("🎯 OUTPUT:")
        final_key = "unet_likelihood/unet_likelihood/kernel:0"
        if final_key in self.synthseg_weights:
            shape = self.synthseg_weights[final_key].shape
            n_classes = shape[4]
            layers.append(f"  Conv3D: 24→{n_classes}, kernel=1x1x1")
            layers.append(f"  Softmax(dim=-1)")
            layers.append(f"  OUTPUT: [B, H, W, D, {n_classes}]")
        
        return layers
    
    def _parse_pytorch_layers(self) -> List[str]:
        """Parse PyTorch architecture from model"""
        layers = []
        
        # Input
        layers.append("🔵 INPUT: [B, 1, D, H, W]")
        layers.append("")
        
        # Encoder
        layers.append("🔽 ENCODER (encoder.downs):")
        encoder_channels = [24, 48, 96, 192, 384]
        
        for level, ch in enumerate(encoder_channels):
            layers.append(f"  downs[{level}] (→{ch} channels):")
            
            # Determine input channels
            in_ch = 1 if level == 0 else encoder_channels[level-1]
            
            layers.append(f"    Conv3d: {in_ch}→{ch}, kernel=3, bias=False")
            layers.append(f"    InstanceNorm3d({ch})")
            layers.append(f"    ELU(inplace=True)")
            layers.append(f"    Conv3d: {ch}→{ch}, kernel=3, bias=False")
            layers.append(f"    InstanceNorm3d({ch})")
            layers.append(f"    ELU(inplace=True)")
            
            if level < len(encoder_channels) - 1:
                layers.append(f"    MaxPool3d(2)")
            layers.append("")
        
        # Decoder
        layers.append("🔼 DECODER (seg_head):")
        decoder_channels = [384, 192, 96, 48, 24]
        
        for level in range(len(decoder_channels)-1):
            in_ch = decoder_channels[level]
            out_ch = decoder_channels[level+1]
            layers.append(f"  ups[{level}] + convs[{level}] (→{out_ch} channels):")
            layers.append(f"    ConvTranspose3d: {in_ch}→{out_ch}, kernel=2, stride=2")
            layers.append(f"    Concatenate with skip (→{out_ch*2} channels)")
            layers.append(f"    Conv3d: {out_ch*2}→{out_ch}, kernel=3, bias=False")
            layers.append(f"    InstanceNorm3d({out_ch})")
            layers.append(f"    ELU(inplace=True)")
            layers.append(f"    Conv3d: {out_ch}→{out_ch}, kernel=3, bias=False")
            layers.append(f"    InstanceNorm3d({out_ch})")
            layers.append(f"    ELU(inplace=True)")
            layers.append("")
        
        # Get actual n_classes from model
        final_weight_key = "seg_head.out.weight"
        if final_weight_key in self.pytorch_state_dict:
            n_classes = self.pytorch_state_dict[final_weight_key].shape[0]
        else:
            n_classes = "N"
        
        layers.append("🎯 SEGMENTATION OUTPUT:")
        layers.append(f"  Conv3d: 24→{n_classes}, kernel=1")
        layers.append(f"  OUTPUT: [B, {n_classes}, D, H, W]")
        layers.append("")
        
        layers.append("🧠 AGE PREDICTION HEAD:")
        layers.append("  AdaptiveAvgPool3d(1)")
        layers.append("  Flatten()")
        layers.append("  Linear(encoder_ch + decoder_ch, 256)")
        layers.append("  ReLU(inplace=True)")
        layers.append("  Linear(256, 1)")
        layers.append("  OUTPUT: [B, 1] (age prediction)")
        
        return layers
    
    def _print_detailed_layer_mapping(self):
        """Print detailed layer-by-layer mapping"""
        print("\n" + "="*120)
        print("🔗 DETAILED LAYER MAPPING")
        print("="*120)
        
        # Encoder mapping
        print("\n🔵 ENCODER MAPPING:")
        print("─" * 80)
        
        for level in range(5):
            print(f"\n📁 Level {level}:")
            for conv_idx in range(2):
                synthseg_conv = f"unet_conv_downarm_{level}_{conv_idx}/unet_conv_downarm_{level}_{conv_idx}/kernel:0"
                pytorch_conv = f"encoder.downs.{level}.block.{conv_idx*3}.weight"
                
                status = "✅" if synthseg_conv in self.synthseg_weights else "❌"
                
                print(f"  {status} Conv {conv_idx}: {synthseg_conv}")
                print(f"     → {pytorch_conv}")
                
                if conv_idx == 1:  # Normalization after second conv
                    synthseg_norm = f"unet_bn_down_{level}/unet_bn_down_{level}/gamma:0"
                    pytorch_norm = f"encoder.downs.{level}.block.{conv_idx*3+1}.weight"
                    
                    status = "✅" if synthseg_norm in self.synthseg_weights else "❌"
                    print(f"  {status} Norm: {synthseg_norm}")
                    print(f"     → {pytorch_norm}")
        
        # Decoder mapping
        print("\n🔴 DECODER MAPPING:")
        print("─" * 80)
        
        decoder_mapping = [(5, 0), (6, 1), (7, 2), (8, 3)]
        for synthseg_level, pytorch_level in decoder_mapping:
            print(f"\n📁 Level {synthseg_level} → PyTorch Level {pytorch_level}:")
            
            for conv_idx in range(2):
                synthseg_conv = f"unet_conv_uparm_{synthseg_level}_{conv_idx}/unet_conv_uparm_{synthseg_level}_{conv_idx}/kernel:0"
                pytorch_conv = f"seg_head.convs.{pytorch_level}.block.{conv_idx*3}.weight"
                
                status = "✅" if synthseg_conv in self.synthseg_weights else "❌"
                
                print(f"  {status} Conv {conv_idx}: {synthseg_conv}")
                print(f"     → {pytorch_conv}")
                
                if conv_idx == 1:
                    synthseg_norm = f"unet_bn_up_{pytorch_level}/unet_bn_up_{pytorch_level}/gamma:0"
                    pytorch_norm = f"seg_head.convs.{pytorch_level}.block.{conv_idx*3+1}.weight"
                    
                    status = "✅" if synthseg_norm in self.synthseg_weights else "❌"
                    print(f"  {status} Norm: {synthseg_norm}")
                    print(f"     → {pytorch_norm}")
        
        # Final layer mapping
        print("\n🎯 FINAL LAYER MAPPING:")
        print("─" * 80)
        
        synthseg_final = "unet_likelihood/unet_likelihood/kernel:0"
        pytorch_final = "seg_head.out.weight"
        
        status = "✅" if synthseg_final in self.synthseg_weights else "❌"
        print(f"{status} Final conv: {synthseg_final}")
        print(f"   → {pytorch_final}")
        
        if synthseg_final in self.synthseg_weights:
            synthseg_shape = self.synthseg_weights[synthseg_final].shape
            pytorch_shape = self.pytorch_state_dict[pytorch_final].shape if pytorch_final in self.pytorch_state_dict else "Unknown"
            print(f"   SynthSeg shape: {synthseg_shape} (Keras format)")
            print(f"   PyTorch shape:  {pytorch_shape} (PyTorch format)")
            
            if synthseg_shape[4] != pytorch_shape[0]:
                print(f"   ⚠️  Class mismatch: SynthSeg={synthseg_shape[4]}, PyTorch={pytorch_shape[0]}")
    
    def _print_parameter_statistics(self):
        """Print parameter count statistics"""
        print("\n" + "="*120)
        print("📊 PARAMETER STATISTICS")
        print("="*120)
        
        # Count SynthSeg parameters
        synthseg_params = self._count_synthseg_parameters()
        
        # Count PyTorch parameters
        pytorch_params = self._count_pytorch_parameters()
        
        print(f"\n{'Component':<20} {'SynthSeg':<15} {'PyTorch':<15} {'Status'}")
        print("─" * 70)
        
        for component in ['Encoder', 'Decoder', 'Final', 'Age Head', 'Total']:
            s_count = synthseg_params.get(component, 0)
            p_count = pytorch_params.get(component, 0)
            
            if component == 'Age Head':
                status = "PyTorch only"
            elif s_count == 0:
                status = "Not available"
            elif abs(s_count - p_count) < 1000:
                status = "✅ Similar"
            else:
                status = "⚠️  Different"
            
            print(f"{component:<20} {s_count:>10,}     {p_count:>10,}     {status}")
    
    def _count_synthseg_parameters(self) -> Dict[str, int]:
        """Count parameters in SynthSeg model"""
        counts = defaultdict(int)
        
        for name, weight in self.synthseg_weights.items():
            param_count = np.prod(weight.shape)
            
            if 'downarm' in name:
                counts['Encoder'] += param_count
            elif 'uparm' in name or 'bn_up' in name:
                counts['Decoder'] += param_count
            elif 'bn_down' in name:
                counts['Encoder'] += param_count
            elif 'likelihood' in name:
                counts['Final'] += param_count
            
            counts['Total'] += param_count
        
        return counts
    
    def _count_pytorch_parameters(self) -> Dict[str, int]:
        """Count parameters in PyTorch model"""
        counts = defaultdict(int)
        
        for name, param in self.pytorch_state_dict.items():
            param_count = param.numel()
            
            if 'encoder' in name:
                counts['Encoder'] += param_count
            elif 'seg_head' in name and 'out' not in name:
                counts['Decoder'] += param_count
            elif 'seg_head.out' in name:
                counts['Final'] += param_count
            elif 'age_head' in name:
                counts['Age Head'] += param_count
            
            counts['Total'] += param_count
        
        return counts


class SynthSegWeightTransfer:
    """
    Precise weight transfer from SynthSeg to PyTorch based on inspection results.
    """
    
    def __init__(self, h5_path: str, verbose: bool = True):
        self.h5_path = Path(h5_path)
        self.verbose = verbose
        self.keras_weights = {}
        self.load_keras_weights()
        
    def load_keras_weights(self):
        """Load all weights from the .h5 file"""
        if not self.h5_path.exists():
            raise FileNotFoundError(f"SynthSeg model not found: {self.h5_path}")
            
        with h5py.File(self.h5_path, 'r') as f:
            def extract_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    self.keras_weights[name] = np.array(obj)
            f.visititems(extract_weights)
            
        if self.verbose:
            print(f"✅ Loaded {len(self.keras_weights)} weight tensors from {self.h5_path.name}")
    
    def show_architecture_comparison(self, pytorch_model: nn.Module):
        """Show detailed architecture comparison"""
        visualizer = ArchitectureVisualizer(self.keras_weights, pytorch_model)
        visualizer.print_architecture_comparison()
    
    @staticmethod
    def convert_conv3d_weight(keras_weight: np.ndarray) -> torch.Tensor:
        """Convert Conv3D weights: Keras (D,H,W,Cin,Cout) -> PyTorch (Cout,Cin,D,H,W)"""
        if len(keras_weight.shape) != 5:
            raise ValueError(f"Expected 5D conv weight, got {keras_weight.shape}")
        return torch.from_numpy(np.transpose(keras_weight, (4, 3, 0, 1, 2))).float()
    
    @staticmethod
    def convert_norm_weight(keras_weight: np.ndarray) -> torch.Tensor:
        """Convert normalization weights (1D tensors)"""
        return torch.from_numpy(keras_weight).float()
    
    def transfer_to_pytorch_model(self, pytorch_model: nn.Module) -> Dict[str, bool]:
        """Transfer weights to PyTorch model with exact mapping"""
        transfer_log = {}
        pytorch_state_dict = pytorch_model.state_dict()
        new_state_dict = pytorch_state_dict.copy()
        
        if self.verbose:
            print("\n" + "="*80)
            print("🔄 STARTING WEIGHT TRANSFER")
            print("="*80)
        
        # Transfer encoder weights
        self._transfer_encoder_weights(new_state_dict, transfer_log)
        
        # Transfer decoder weights  
        self._transfer_decoder_weights(new_state_dict, transfer_log)
        
        # Transfer final layer
        self._transfer_final_layer(new_state_dict, transfer_log)
        
        # Load the new state dict
        try:
            pytorch_model.load_state_dict(new_state_dict, strict=False)
            successful = sum(transfer_log.values())
            total = len(transfer_log)
            if self.verbose:
                print(f"\n✅ Weight transfer complete: {successful}/{total} layers transferred")
                self._print_transfer_summary(transfer_log)
        except Exception as e:
            print(f"❌ Error loading state dict: {e}")
            raise
        
        return transfer_log
    
    def _transfer_encoder_weights(self, state_dict: Dict, transfer_log: Dict):
        """Transfer encoder weights with exact SynthSeg mapping"""
        if self.verbose:
            print("\n🔵 Transferring ENCODER weights...")
        
        for level in range(5):  # levels 0-4
            for conv_idx in range(2):  # 2 convs per level
                
                # === CONVOLUTION WEIGHTS ===
                synthseg_conv_key = f"unet_conv_downarm_{level}_{conv_idx}/unet_conv_downarm_{level}_{conv_idx}/kernel:0"
                pytorch_conv_key = f"encoder.downs.{level}.block.{conv_idx*3}.weight"
                
                if synthseg_conv_key in self.keras_weights and pytorch_conv_key in state_dict:
                    try:
                        keras_weight = self.keras_weights[synthseg_conv_key]
                        pytorch_weight = self.convert_conv3d_weight(keras_weight)
                        
                        expected_shape = state_dict[pytorch_conv_key].shape
                        if pytorch_weight.shape == expected_shape:
                            state_dict[pytorch_conv_key] = pytorch_weight
                            transfer_log[pytorch_conv_key] = True
                            if self.verbose:
                                print(f"  ✅ Enc Conv {level}-{conv_idx}: {keras_weight.shape} → {pytorch_weight.shape}")
                        else:
                            transfer_log[pytorch_conv_key] = False
                            if self.verbose:
                                print(f"  ❌ Enc Conv {level}-{conv_idx}: Shape mismatch {pytorch_weight.shape} vs {expected_shape}")
                    except Exception as e:
                        transfer_log[pytorch_conv_key] = False
                        if self.verbose:
                            print(f"  ❌ Enc Conv {level}-{conv_idx}: Error {e}")
                else:
                    transfer_log[pytorch_conv_key] = False
                    if self.verbose:
                        print(f"  ❌ Enc Conv {level}-{conv_idx}: Key not found")
                
                # === NORMALIZATION WEIGHTS ===
                if conv_idx == 1:  # Transfer norm after second conv
                    synthseg_gamma_key = f"unet_bn_down_{level}/unet_bn_down_{level}/gamma:0"
                    synthseg_beta_key = f"unet_bn_down_{level}/unet_bn_down_{level}/beta:0"
                    
                    pytorch_norm_weight_key = f"encoder.downs.{level}.block.{conv_idx*3+1}.weight"
                    pytorch_norm_bias_key = f"encoder.downs.{level}.block.{conv_idx*3+1}.bias"
                    
                    # Transfer gamma → InstanceNorm weight
                    if synthseg_gamma_key in self.keras_weights and pytorch_norm_weight_key in state_dict:
                        try:
                            gamma_weight = self.convert_norm_weight(self.keras_weights[synthseg_gamma_key])
                            state_dict[pytorch_norm_weight_key] = gamma_weight
                            transfer_log[pytorch_norm_weight_key] = True
                            if self.verbose:
                                print(f"  ✅ Enc Norm {level} γ→weight: {gamma_weight.shape}")
                        except Exception as e:
                            transfer_log[pytorch_norm_weight_key] = False
                            if self.verbose:
                                print(f"  ❌ Enc Norm {level} γ: {e}")
                    else:
                        transfer_log[pytorch_norm_weight_key] = False
                    
                    # Transfer beta → InstanceNorm bias
                    if synthseg_beta_key in self.keras_weights and pytorch_norm_bias_key in state_dict:
                        try:
                            beta_weight = self.convert_norm_weight(self.keras_weights[synthseg_beta_key])
                            state_dict[pytorch_norm_bias_key] = beta_weight
                            transfer_log[pytorch_norm_bias_key] = True
                            if self.verbose:
                                print(f"  ✅ Enc Norm {level} β→bias: {beta_weight.shape}")
                        except Exception as e:
                            transfer_log[pytorch_norm_bias_key] = False
                            if self.verbose:
                                print(f"  ❌ Enc Norm {level} β: {e}")
                    else:
                        transfer_log[pytorch_norm_bias_key] = False
    
    def _transfer_decoder_weights(self, state_dict: Dict, transfer_log: Dict):
        """Transfer decoder weights with exact SynthSeg mapping"""
        if self.verbose:
            print("\n🔴 Transferring DECODER weights...")
        
        # SynthSeg uparm levels 5-8 map to PyTorch levels 0-3
        decoder_mapping = [(5, 0), (6, 1), (7, 2), (8, 3)]
        
        for synthseg_level, pytorch_level in decoder_mapping:
            for conv_idx in range(2):
                
                # === DECODER CONVOLUTION WEIGHTS ===
                synthseg_conv_key = f"unet_conv_uparm_{synthseg_level}_{conv_idx}/unet_conv_uparm_{synthseg_level}_{conv_idx}/kernel:0"
                pytorch_conv_key = f"seg_head.convs.{pytorch_level}.block.{conv_idx*3}.weight"
                
                if synthseg_conv_key in self.keras_weights and pytorch_conv_key in state_dict:
                    try:
                        keras_weight = self.keras_weights[synthseg_conv_key]
                        pytorch_weight = self.convert_conv3d_weight(keras_weight)
                        
                        expected_shape = state_dict[pytorch_conv_key].shape
                        if pytorch_weight.shape == expected_shape:
                            state_dict[pytorch_conv_key] = pytorch_weight
                            transfer_log[pytorch_conv_key] = True
                            if self.verbose:
                                print(f"  ✅ Dec Conv {synthseg_level}-{conv_idx}: {keras_weight.shape} → {pytorch_weight.shape}")
                        else:
                            transfer_log[pytorch_conv_key] = False
                            if self.verbose:
                                print(f"  ❌ Dec Conv {synthseg_level}-{conv_idx}: Shape mismatch {pytorch_weight.shape} vs {expected_shape}")
                    except Exception as e:
                        transfer_log[pytorch_conv_key] = False
                        if self.verbose:
                            print(f"  ❌ Dec Conv {synthseg_level}-{conv_idx}: Error {e}")
                else:
                    transfer_log[pytorch_conv_key] = False
                    if self.verbose:
                        print(f"  ❌ Dec Conv {synthseg_level}-{conv_idx}: Key not found")
                
                # === DECODER NORMALIZATION WEIGHTS ===
                if conv_idx == 1:  # Transfer norm after second conv
                    bn_level = pytorch_level  # bn_up levels 0-3 map to pytorch levels 0-3
                    
                    synthseg_gamma_key = f"unet_bn_up_{bn_level}/unet_bn_up_{bn_level}/gamma:0"
                    synthseg_beta_key = f"unet_bn_up_{bn_level}/unet_bn_up_{bn_level}/beta:0"
                    
                    pytorch_norm_weight_key = f"seg_head.convs.{pytorch_level}.block.{conv_idx*3+1}.weight"
                    pytorch_norm_bias_key = f"seg_head.convs.{pytorch_level}.block.{conv_idx*3+1}.bias"
                    
                    # Transfer gamma → InstanceNorm weight
                    if synthseg_gamma_key in self.keras_weights and pytorch_norm_weight_key in state_dict:
                        try:
                            gamma_weight = self.convert_norm_weight(self.keras_weights[synthseg_gamma_key])
                            state_dict[pytorch_norm_weight_key] = gamma_weight
                            transfer_log[pytorch_norm_weight_key] = True
                            if self.verbose:
                                print(f"  ✅ Dec Norm {bn_level} γ→weight: {gamma_weight.shape}")
                        except Exception as e:
                            transfer_log[pytorch_norm_weight_key] = False
                            if self.verbose:
                                print(f"  ❌ Dec Norm {bn_level} γ: {e}")
                    else:
                        transfer_log[pytorch_norm_weight_key] = False
                    
                    # Transfer beta → InstanceNorm bias
                    if synthseg_beta_key in self.keras_weights and pytorch_norm_bias_key in state_dict:
                        try:
                            beta_weight = self.convert_norm_weight(self.keras_weights[synthseg_beta_key])
                            state_dict[pytorch_norm_bias_key] = beta_weight
                            transfer_log[pytorch_norm_bias_key] = True
                            if self.verbose:
                                print(f"  ✅ Dec Norm {bn_level} β→bias: {beta_weight.shape}")
                        except Exception as e:
                            transfer_log[pytorch_norm_bias_key] = False
                            if self.verbose:
                                print(f"  ❌ Dec Norm {bn_level} β: {e}")
                    else:
                        transfer_log[pytorch_norm_bias_key] = False
        
        # Skip upsampling layers (different architecture)
        for level in range(4):
            ups_key = f"seg_head.ups.{level}.weight"
            if ups_key in self.pytorch_model.state_dict():
                transfer_log[ups_key] = False
                if self.verbose and level == 0:
                    print(f"  ⚠️  Upsampling layers use random init (architecture difference)")
    
    def _transfer_final_layer(self, state_dict: Dict, transfer_log: Dict):
        """Transfer final segmentation layer"""
        if self.verbose:
            print("\n🟡 Transferring FINAL LAYER...")
        
        synthseg_final_kernel = "unet_likelihood/unet_likelihood/kernel:0"
        synthseg_final_bias = "unet_likelihood/unet_likelihood/bias:0"
        
        pytorch_final_weight = "seg_head.out.weight"
        pytorch_final_bias = "seg_head.out.bias"
        
        # Transfer final conv weight
        if synthseg_final_kernel in self.keras_weights and pytorch_final_weight in state_dict:
            try:
                keras_weight = self.keras_weights[synthseg_final_kernel]
                pytorch_weight = self.convert_conv3d_weight(keras_weight)
                
                expected_shape = state_dict[pytorch_final_weight].shape
                if pytorch_weight.shape == expected_shape:
                    state_dict[pytorch_final_weight] = pytorch_weight
                    transfer_log[pytorch_final_weight] = True
                    if self.verbose:
                        print(f"  ✅ Final conv weight: {keras_weight.shape} → {pytorch_weight.shape}")
                else:
                    transfer_log[pytorch_final_weight] = False
                    if self.verbose:
                        print(f"  ⚠️  Final conv: Shape mismatch {pytorch_weight.shape} vs {expected_shape}")
                        print(f"      SynthSeg: {keras_weight.shape[-1]} classes, PyTorch: {expected_shape[0]} classes")
                        print(f"      Using random initialization for final layer")
            except Exception as e:
                transfer_log[pytorch_final_weight] = False
                if self.verbose:
                    print(f"  ❌ Final conv weight: {e}")
        else:
            transfer_log[pytorch_final_weight] = False
        
        # Transfer final conv bias
        if synthseg_final_bias in self.keras_weights and pytorch_final_bias in state_dict:
            try:
                keras_bias = self.keras_weights[synthseg_final_bias]
                pytorch_bias = self.convert_norm_weight(keras_bias)
                
                expected_shape = state_dict[pytorch_final_bias].shape
                if pytorch_bias.shape == expected_shape:
                    state_dict[pytorch_final_bias] = pytorch_bias
                    transfer_log[pytorch_final_bias] = True
                    if self.verbose:
                        print(f"  ✅ Final conv bias: {keras_bias.shape} → {pytorch_bias.shape}")
                else:
                    transfer_log[pytorch_final_bias] = False
                    if self.verbose:
                        print(f"  ⚠️  Final conv bias: Shape mismatch")
            except Exception as e:
                transfer_log[pytorch_final_bias] = False
                if self.verbose:
                    print(f"  ❌ Final conv bias: {e}")
        else:
            transfer_log[pytorch_final_bias] = False
    
    def _print_transfer_summary(self, transfer_log: Dict):
        """Print detailed transfer summary"""
        successful = [k for k, v in transfer_log.items() if v]
        failed = [k for k, v in transfer_log.items() if not v]
        
        print(f"\n📊 TRANSFER SUMMARY:")
        print(f"✅ Successfully transferred: {len(successful)}")
        print(f"❌ Failed/Skipped: {len(failed)}")
        
        # Component breakdown
        encoder_success = [k for k in successful if 'encoder' in k]
        decoder_success = [k for k in successful if 'seg_head' in k and 'out' not in k]
        final_success = [k for k in successful if 'seg_head.out' in k]
        ups_failed = [k for k in failed if 'ups' in k]
        age_head = [k for k in transfer_log.keys() if 'age_head' in k]
        
        print(f"\n📈 Component breakdown:")
        print(f"  🔵 Encoder: {len(encoder_success)} layers transferred")
        print(f"  🔴 Decoder: {len(decoder_success)} layers transferred")
        print(f"  🟡 Final layer: {len(final_success)} layers transferred")
        print(f"  ⚪ Upsampling: {len(ups_failed)} layers using random init (expected)")
        print(f"  🧠 Age head: {len(age_head)} layers using random init (expected)")
        
        if self.verbose:
            failed_other = [k for k in failed if 'ups' not in k and 'age_head' not in k]
            if failed_other:
                print(f"\n⚠️  Unexpected failures:")
                for layer in sorted(failed_other):
                    print(f"   • {layer}")


def load_synthseg_pretrained_weights(model: nn.Module, synthseg_path: str, 
                                   show_architecture: bool = True,
                                   verbose: bool = True) -> nn.Module:
    """
    Load SynthSeg weights into PyTorch model with architecture visualization
    
    Args:
        model: PyTorch MultiTaskBrainAge model
        synthseg_path: Path to SynthSeg .h5 file
        show_architecture: Whether to show detailed architecture comparison
        verbose: Print detailed transfer info
    
    Returns:
        Model with transferred weights
    """
    if not Path(synthseg_path).exists():
        print(f"⚠️  SynthSeg weights not found at {synthseg_path}")
        print("   Using random initialization for all layers")
        return model
    
    try:
        print(f"\n🚀 Loading SynthSeg pretrained weights from {Path(synthseg_path).name}")
        
        transfer = SynthSegWeightTransfer(synthseg_path, verbose=verbose)
        
        if show_architecture:
            transfer.show_architecture_comparison(model)
        
        transfer_log = transfer.transfer_to_pytorch_model(model)
        
        # Calculate success rates
        total = len(transfer_log)
        successful = sum(transfer_log.values())
        success_rate = successful / total if total > 0 else 0
        
        # Component success rates
        encoder_layers = [k for k in transfer_log.keys() if 'encoder' in k]
        encoder_success = sum(transfer_log[k] for k in encoder_layers)
        encoder_rate = encoder_success / len(encoder_layers) if encoder_layers else 0
        
        decoder_layers = [k for k in transfer_log.keys() if 'seg_head' in k and 'out' not in k and 'ups' not in k]
        decoder_success = sum(transfer_log[k] for k in decoder_layers)
        decoder_rate = decoder_success / len(decoder_layers) if decoder_layers else 0
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Overall: {success_rate:.1%} ({successful}/{total} layers)")
        print(f"   Encoder: {encoder_rate:.1%} ({encoder_success}/{len(encoder_layers)} layers)")
        print(f"   Decoder: {decoder_rate:.1%} ({decoder_success}/{len(decoder_layers)} layers)")
        print(f"   ✨ Model ready for training with SynthSeg features!")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to transfer SynthSeg weights: {e}")
        print("   Continuing with random initialization...")
        import traceback
        if verbose:
            traceback.print_exc()
        return model


# Utility functions for standalone usage
def inspect_synthseg_model(h5_path: str):
    """Standalone function to inspect SynthSeg model structure"""
    print(f"Inspecting SynthSeg model: {h5_path}")
    print("="*80)
    
    transfer = SynthSegWeightTransfer(h5_path, verbose=True)
    
    # Create a dummy PyTorch model for comparison
    try:
        from brain_age_pred.models.multi_head import MultiTaskBrainAge
        dummy_model = MultiTaskBrainAge(n_classes=33)  # SynthSeg has 33 classes
        transfer.show_architecture_comparison(dummy_model)
    except ImportError:
        print("Could not import MultiTaskBrainAge model for comparison")
        print("Available SynthSeg weights:")
        for name, weight in transfer.keras_weights.items():
            if 'kernel' in name or 'gamma' in name or 'beta' in name:
                print(f"  {name}: {weight.shape}")


if __name__ == "__main__":
    # Example usage
    synthseg_path = "/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/OtherRepos/SynthSeg/models/synthseg_2.0.h5"
    
    # Option 1: Just inspect the model
    inspect_synthseg_model(synthseg_path)
    
    # Option 2: Transfer weights (uncomment to use)
    # from brain_age_pred.models.multi_head import MultiTaskBrainAge
    # model = MultiTaskBrainAge(n_classes=33)
    # model = load_synthseg_pretrained_weights(model, synthseg_path, show_architecture=True)