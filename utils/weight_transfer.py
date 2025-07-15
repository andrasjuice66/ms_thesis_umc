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
        """
        Initialize the weight transfer utility.
        
        Args:
            h5_path: Path to SynthSeg .h5 model file
            torch_model: Your PyTorch multi-task model
        """
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
                    # Convert to numpy array
                    weights_dict[name] = np.array(obj)
                    
            f.visititems(collect_weights)
            
        return weights_dict
    
    def map_encoder_weights(self, h5_weights: Dict) -> Dict:
        """
        Map SynthSeg encoder weights to PyTorch encoder.
        
        SynthSeg naming pattern (example):
        'unet_conv_downarm_0_0/kernel:0' -> encoder.downs[0].block[0].weight
        'unet_conv_downarm_0_0/bias:0' -> encoder.downs[0].block[0].bias
        """
        mapping = {}
        
        for h5_name, weight in h5_weights.items():
            # Parse SynthSeg layer names
            if 'conv_downarm' in h5_name:
                # Extract level and conv indices
                match = re.search(r'conv_downarm_(\d+)_(\d+)', h5_name)
                if match:
                    level, conv_idx = int(match.group(1)), int(match.group(2))
                    
                    if 'kernel' in h5_name:
                        # Map to PyTorch conv weight
                        torch_name = f'encoder.downs.{level}.block.{conv_idx * 3}.weight'
                        # Transpose from TF format (H,W,D,C_in,C_out) to PyTorch (C_out,C_in,D,H,W)
                        if len(weight.shape) == 5:
                            weight = np.transpose(weight, (4, 3, 2, 0, 1))
                        mapping[torch_name] = torch.from_numpy(weight)
                        
                    elif 'bias' in h5_name:
                        torch_name = f'encoder.downs.{level}.block.{conv_idx * 3}.bias'
                        mapping[torch_name] = torch.from_numpy(weight)
                        
        return mapping
    
    def map_decoder_weights(self, h5_weights: Dict) -> Dict:
        """
        Map SynthSeg decoder weights to PyTorch decoder.
        
        SynthSeg naming pattern:
        'unet_conv_uparm_5_0/kernel:0' -> seg_head.convs[0].block[0].weight
        """
        mapping = {}
        
        for h5_name, weight in h5_weights.items():
            if 'conv_uparm' in h5_name:
                match = re.search(r'conv_uparm_(\d+)_(\d+)', h5_name)
                if match:
                    level, conv_idx = int(match.group(1)), int(match.group(2))
                    # Map decoder levels (SynthSeg levels 5,6,7,8 -> PyTorch indices 0,1,2,3)
                    decoder_idx = level - 5  # Adjust based on your architecture
                    
                    if decoder_idx >= 0 and 'kernel' in h5_name:
                        torch_name = f'seg_head.convs.{decoder_idx}.block.{conv_idx * 3}.weight'
                        if len(weight.shape) == 5:
                            weight = np.transpose(weight, (4, 3, 2, 0, 1))
                        mapping[torch_name] = torch.from_numpy(weight)
                        
                    elif decoder_idx >= 0 and 'bias' in h5_name:
                        torch_name = f'seg_head.convs.{decoder_idx}.block.{conv_idx * 3}.bias'
                        mapping[torch_name] = torch.from_numpy(weight)
        
        # Map final segmentation layer
        for h5_name, weight in h5_weights.items():
            if 'unet_likelihood' in h5_name or 'prediction' in h5_name:
                if 'kernel' in h5_name:
                    if len(weight.shape) == 5:
                        weight = np.transpose(weight, (4, 3, 2, 0, 1))
                    mapping['seg_head.out.weight'] = torch.from_numpy(weight)
                elif 'bias' in h5_name:
                    mapping['seg_head.out.bias'] = torch.from_numpy(weight)
                    
        return mapping
    
    def map_normalization_weights(self, h5_weights: Dict) -> Dict:
        """Map batch normalization or instance normalization weights."""
        mapping = {}
        
        for h5_name, weight in h5_weights.items():
            # Look for normalization layers
            if any(norm_type in h5_name.lower() for norm_type in ['batch_norm', 'instance_norm', 'bn']):
                # Extract layer information and map to corresponding PyTorch norm layers
                # This will depend on your specific normalization setup
                pass
                
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
        
        # Freeze segmentation layers if desired (for fine-tuning)
        # self.freeze_segmentation_layers()
        
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
    
    def freeze_segmentation_layers(self):
        """Freeze segmentation-related layers for brain age fine-tuning."""
        frozen_layers = []
        
        for name, param in self.torch_model.named_parameters():
            if any(seg_part in name for seg_part in ['encoder', 'seg_head']):
                param.requires_grad = False
                frozen_layers.append(name)
                
        print(f"Frozen {len(frozen_layers)} segmentation layers")
        return frozen_layers
    
    def unfreeze_layers(self, layer_patterns: List[str]):
        """Unfreeze specific layer patterns for fine-tuning."""
        unfrozen = []
        
        for name, param in self.torch_model.named_parameters():
            if any(pattern in name for pattern in layer_patterns):
                param.requires_grad = True
                unfrozen.append(name)
                
        print(f"Unfrozen {len(unfrozen)} layers matching patterns: {layer_patterns}")
        return unfrozen
    
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
            for name, reason in list(summary['skipped'].items())[:10]:  # Show first 10
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
    
    Args:
        h5_path: Path to SynthSeg .h5 file
        torch_model: PyTorch model to transfer weights to
        transfer_encoder: Whether to transfer encoder weights
        transfer_decoder: Whether to transfer decoder weights
        freeze_seg_layers: Whether to freeze segmentation layers after transfer
    
    Returns:
        Transfer summary dictionary
    """
    transfer = SynthSegToTorchTransfer(h5_path, torch_model)
    summary = transfer.transfer_weights(transfer_encoder, transfer_decoder)
    
    if freeze_seg_layers:
        transfer.freeze_segmentation_layers()
    
    return summary


# Example usage function
def create_transfer_script(h5_path: str, model_checkpoint_path: Optional[str] = None):
    """Create a script to perform the weight transfer."""
    script_content = f'''
#!/usr/bin/env python3
"""
Script to transfer SynthSeg weights to PyTorch model.
Generated automatically.
"""

import torch
from brain_age_pred.models.multi_head import MultiTaskBrainAge
from brain_age_pred.utils.weight_transfer import transfer_synthseg_weights

# Load your PyTorch model
model = MultiTaskBrainAge(n_classes=35)  # Adjust n_classes as needed

# Load existing checkpoint if available
if "{model_checkpoint_path}" and Path("{model_checkpoint_path}").exists():
    checkpoint = torch.load("{model_checkpoint_path}", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Loaded existing PyTorch checkpoint")

# Transfer SynthSeg weights
transfer_summary = transfer_synthseg_weights(
    h5_path="{h5_path}",
    torch_model=model,
    transfer_encoder=True,
    transfer_decoder=True,
    freeze_seg_layers=False  # Set to True if you want to freeze seg layers
)

# Save the model with transferred weights
torch.save({{
    'model_state_dict': model.state_dict(),
    'transfer_summary': transfer_summary
}}, 'model_with_synthseg_weights.pth')

print("Weight transfer complete! Saved to 'model_with_synthseg_weights.pth'")
'''
    
    with open('transfer_synthseg_weights.py', 'w') as f:
        f.write(script_content)
    
    print("Created transfer script: transfer_synthseg_weights.py") 