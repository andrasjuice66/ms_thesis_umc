#!/usr/bin/env python3
"""
Transfer SynthSeg weights to PyTorch multi-task model.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.models.multi_head import MultiTaskBrainAge
from brain_age_pred.utils.weight_transfer import transfer_synthseg_weights
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES

# Update your weight transfer utility
import h5py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re


class SynthSegToTorchTransfer:
    """Transfer SynthSeg weights from Keras/TF .h5 format to PyTorch model."""
    
    def __init__(self, h5_path: str, torch_model: nn.Module):
        self.h5_path = Path(h5_path)
        self.torch_model = torch_model
        self.weight_mapping = {}
        self.transfer_log = []
        
    def load_h5_weights(self) -> Dict:
        """Load weights from .h5 file and organize them."""
        weights_dict = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            def collect_weights(name, obj):
                if hasattr(obj, 'shape') and obj.shape:
                    weights_dict[name] = np.array(obj)
                    
            f.visititems(collect_weights)
            
        return weights_dict
    
    def map_encoder_weights(self, h5_weights: Dict) -> Dict:
        """Fixed encoder mapping - no bias for conv layers."""
        mapping = {}
        
        for level in range(5):  # levels 0-4
            for conv_idx in range(2):  # 0, 1 (two convs per level)
                
                h5_kernel = f'unet_conv_downarm_{level}_{conv_idx}/unet_conv_downarm_{level}_{conv_idx}/kernel:0'
                
                if h5_kernel in h5_weights:
                    # Map to correct PyTorch conv indices (0 and 3)
                    torch_conv_idx = 0 if conv_idx == 0 else 3
                    torch_conv_name = f'encoder.downs.{level}.block.{torch_conv_idx}.weight'
                    
                    kernel = h5_weights[h5_kernel]
                    kernel_pt = np.transpose(kernel, (4, 3, 2, 0, 1))
                    mapping[torch_conv_name] = torch.from_numpy(kernel_pt)
        
        return mapping
    
    def map_decoder_weights(self, h5_weights: Dict) -> Dict:
        """Fixed decoder mapping - only conv weights, skip ConvTranspose3d."""
        mapping = {}
        
        # Only map the ConvBlock layers, skip ConvTranspose3d (ups layers)
        for synthseg_level in range(5, 9):  # 5, 6, 7, 8
            pytorch_idx = synthseg_level - 5  # 0, 1, 2, 3
            
            for conv_idx in range(2):  # 0, 1
                h5_kernel = f'unet_conv_uparm_{synthseg_level}_{conv_idx}/unet_conv_uparm_{synthseg_level}_{conv_idx}/kernel:0'
                
                if h5_kernel in h5_weights:
                    # Map to correct PyTorch conv indices (0 and 3)
                    torch_conv_idx = 0 if conv_idx == 0 else 3
                    torch_conv_name = f'seg_head.convs.{pytorch_idx}.block.{torch_conv_idx}.weight'
                    
                    kernel = h5_weights[h5_kernel]
                    kernel_pt = np.transpose(kernel, (4, 3, 2, 0, 1))
                    mapping[torch_conv_name] = torch.from_numpy(kernel_pt)
        
        # Map final segmentation layer
        h5_final_kernel = 'unet_likelihood/unet_likelihood/kernel:0'
        h5_final_bias = 'unet_likelihood/unet_likelihood/bias:0'
        
        if h5_final_kernel in h5_weights:
            kernel = h5_weights[h5_final_kernel]
            kernel_pt = np.transpose(kernel, (4, 3, 2, 0, 1))
            
            mapping['seg_head.out.weight'] = torch.from_numpy(kernel_pt)
            mapping['seg_head.out.bias'] = torch.from_numpy(h5_weights[h5_final_bias])
        
        return mapping
    
    def transfer_weights(self, 
                        transfer_encoder: bool = True, 
                        transfer_decoder: bool = True,
                        strict: bool = False) -> Dict:
        """
        Perform the weight transfer.
        
        Args:
            transfer_encoder: Whether to transfer encoder weights
            transfer_decoder: Whether to transfer decoder weights  
            strict: Whether to require exact shape matches
        """
        print(f"Loading weights from {self.h5_path}")
        h5_weights = self.load_h5_weights()
        print(f"Found {len(h5_weights)} weight tensors in .h5 file")
        
        # Create weight mappings
        all_mappings = {}
        
        if transfer_encoder:
            encoder_mapping = self.map_encoder_weights(h5_weights)
            all_mappings.update(encoder_mapping)
            print(f"Mapped {len(encoder_mapping)} encoder weights")
            
        if transfer_decoder:
            decoder_mapping = self.map_decoder_weights(h5_weights)
            all_mappings.update(decoder_mapping)
            print(f"Mapped {len(decoder_mapping)} decoder weights")
        
        # Apply weights to PyTorch model
        model_dict = self.torch_model.state_dict()
        transferred = {}
        skipped = {}
        
        for torch_name, weight_tensor in all_mappings.items():
            if torch_name in model_dict:
                if model_dict[torch_name].shape == weight_tensor.shape:
                    model_dict[torch_name] = weight_tensor
                    transferred[torch_name] = weight_tensor.shape
                else:
                    shape_mismatch = f"Expected {model_dict[torch_name].shape}, got {weight_tensor.shape}"
                    skipped[torch_name] = shape_mismatch
                    if strict:
                        raise ValueError(f"Shape mismatch for {torch_name}: {shape_mismatch}")
            else:
                skipped[torch_name] = "Layer not found in PyTorch model"
        
        # Load the updated weights
        self.torch_model.load_state_dict(model_dict, strict=False)
        
        transfer_summary = {
            'transferred': transferred,
            'skipped': skipped,
            'transfer_stats': {
                'total_attempted': len(all_mappings),
                'successfully_transferred': len(transferred),
                'skipped_count': len(skipped)
            }
        }
        
        self.print_transfer_summary(transfer_summary)
        return transfer_summary
    
    def print_transfer_summary(self, summary: Dict):
        """Print a detailed summary of the weight transfer."""
        stats = summary['transfer_stats']
        print("\n" + "="*60)
        print("WEIGHT TRANSFER SUMMARY")
        print("="*60)
        print(f"Total weights attempted: {stats['total_attempted']}")
        print(f"Successfully transferred: {stats['successfully_transferred']}")
        print(f"Skipped: {stats['skipped_count']}")
        print(f"Transfer rate: {stats['successfully_transferred']/stats['total_attempted']*100:.1f}%")
        
        if summary['skipped']:
            print("\nSKIPPED WEIGHTS:")
            for name, reason in list(summary['skipped'].items())[:10]:
                print(f"  {name}: {reason}")
            if len(summary['skipped']) > 10:
                print(f"  ... and {len(summary['skipped'])-10} more")
                
        print("="*60)


def transfer_synthseg_weights(h5_path: str, 
                            torch_model: nn.Module,
                            transfer_encoder: bool = True,
                            transfer_decoder: bool = True,
                            freeze_seg_layers: bool = False) -> Dict:
    """
    Convenience function for SynthSeg weight transfer.
    """
    transfer = SynthSegToTorchTransfer(h5_path, torch_model)
    summary = transfer.transfer_weights(transfer_encoder, transfer_decoder)
    
    if freeze_seg_layers:
        # Freeze segmentation layers
        for name, param in torch_model.named_parameters():
            if any(seg_part in name for seg_part in ['encoder', 'seg_head']):
                param.requires_grad = False
    
    return summary


# Main execution script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    
    from brain_age_pred.models.multi_head import MultiTaskBrainAge
    
    # Configuration
    synthseg_h5_path = "/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/OtherRepos/SynthSeg/models/synthseg_1.0.h5"
    output_model_path = "/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/brain_age_pred/models/synthseg.pth"
    n_classes = 32  # Change from 33 to 32
    
    print("=" * 60)
    print("SynthSeg to PyTorch Weight Transfer")
    print("=" * 60)
    
    # Initialize model with correct number of classes
    print(f"🏗️ Initializing PyTorch model with {n_classes} classes...")
    model = MultiTaskBrainAge(n_classes=n_classes)
    
    # Perform transfer
    print(f"\n🔄 Starting weight transfer...")
    try:
        summary = transfer_synthseg_weights(
            h5_path=synthseg_h5_path,
            torch_model=model,
            transfer_encoder=True,
            transfer_decoder=True,
            freeze_seg_layers=False
        )
        
        # Save model
        print(f"\n💾 Saving model to {output_model_path}...")
        Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'transfer_summary': summary,
            'model_config': {'n_classes': n_classes}
        }, output_model_path)
        
        print("✅ Weight transfer completed successfully!")
        
    except Exception as e:
        print(f"❌ Weight transfer failed: {e}")
        import traceback
        traceback.print_exc() 