#!/usr/bin/env python3
"""
Verify that the remapped checkpoint produces identical outputs to the original checkpoint.
This ensures the weight mapping was done correctly.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.models.brainagenext import BrainAgeNeXt


class BrainAgeNeXtOld(nn.Module):
    """
    OLD architecture that created the original checkpoint.
    This has the nn.Flatten() INSIDE the Sequential.
    """
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.0,
        model_id: str = 'B',
        kernel_size: int = 3,
        deep_supervision: bool = True,
        feature_size: int = 512,
        hidden_size: int = 64
    ):
        super(BrainAgeNeXtOld, self).__init__()
        
        from brain_age_pred.models.create_mednext_encoder_v1 import create_mednext_encoder_v1
        
        self.feature_extractor = create_mednext_encoder_v1(
            num_input_channels=in_channels, 
            num_classes=1, 
            model_id=model_id, 
            kernel_size=kernel_size, 
            deep_supervision=deep_supervision
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regression_head = nn.Sequential(
            nn.Flatten(),  # This was layer 0 (no parameters)
            nn.Linear(feature_size, hidden_size),  # This was layer 1
            nn.ReLU(),  # This was layer 2
            nn.Dropout(dropout_rate),  # This was layer 3
            nn.Linear(hidden_size, 1)  # This was layer 4
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        x = self.global_avg_pool(features)
        age_estimate = self.regression_head(x)
        return age_estimate.squeeze()


def load_checkpoint_with_mapping(checkpoint_path):
    """Load checkpoint and apply key mapping."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            old_state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            old_state_dict = checkpoint['model_state_dict']
        else:
            old_state_dict = checkpoint
    else:
        old_state_dict = checkpoint
    
    # Remove 'module.' prefix if exists
    clean_state_dict = {}
    for k, v in old_state_dict.items():
        clean_key = k[7:] if k.startswith('module.') else k
        clean_state_dict[clean_key] = v
    
    return clean_state_dict


def apply_key_mapping(state_dict):
    """Apply key mapping from old to new architecture."""
    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = old_key
        # Map feature_extractor.* to mednextv1.*
        if old_key.startswith('feature_extractor.'):
            new_key = old_key.replace('feature_extractor.', 'mednextv1.', 1)
        # Map regression_head.* to regression_fc.* with index adjustment
        elif old_key.startswith('regression_head.'):
            new_key = old_key.replace('regression_head.', 'regression_fc.', 1)
            # Fix index shift: 1→0, 4→3
            new_key = new_key.replace('regression_fc.1.', 'regression_fc.0.')
            new_key = new_key.replace('regression_fc.4.', 'regression_fc.3.')
        new_state_dict[new_key] = value
    return new_state_dict


def main():
    print("="*80)
    print("CHECKPOINT EQUIVALENCE VERIFICATION")
    print("="*80)
    
    # Paths
    old_checkpoint_path = "/mnt/c/Projects/thesis_project/checkpoints_best_mae/brainagenext_seg_map/brainagenext_seg_map_best_mae.pt"
    remapped_checkpoint_path = "/mnt/c/Projects/thesis_project/checkpoints_best_mae/brainagenext_seg_map/brainagenext_seg_map_best_mae_remapped.pt"
    
    # Model config (from your YAML)
    model_config = {
        'in_channels': 15,
        'dropout_rate': 0.2,
        'model_id': 'B',
        'kernel_size': 3,
        'deep_supervision': True,
        'feature_size': 512,
        'hidden_size': 64
    }
    
    print("\n1. Loading old checkpoint into old architecture...")
    old_state_dict = load_checkpoint_with_mapping(old_checkpoint_path)
    model_old = BrainAgeNeXtOld(**model_config)
    result = model_old.load_state_dict(old_state_dict, strict=False)
    print(f"   Loaded {len(old_state_dict)} keys")
    print(f"   Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    model_old.eval()
    
    print("\n2. Loading remapped checkpoint into new architecture...")
    remapped_checkpoint = torch.load(remapped_checkpoint_path, map_location='cpu')
    if isinstance(remapped_checkpoint, dict):
        if 'state_dict' in remapped_checkpoint:
            remapped_state_dict = remapped_checkpoint['state_dict']
        elif 'model_state_dict' in remapped_checkpoint:
            remapped_state_dict = remapped_checkpoint['model_state_dict']
        else:
            remapped_state_dict = remapped_checkpoint
    else:
        remapped_state_dict = remapped_checkpoint
    model_new = BrainAgeNeXt(**model_config)
    result = model_new.load_state_dict(remapped_state_dict, strict=False)
    print(f"   Loaded {len(remapped_state_dict)} keys")
    print(f"   Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    model_new.eval()
    
    print("\n3. Creating test input (batch=2, channels=15, size=128x128x128)...")
    # Create a reproducible random input
    torch.manual_seed(42)
    test_input = torch.randn(2, 15, 128, 128, 128)
    
    print("\n4. Running inference with both models...")
    with torch.no_grad():
        output_old = model_old(test_input)
        output_new = model_new(test_input)
    
    print("\n5. Comparing outputs...")
    print(f"   Old model output: {output_old}")
    print(f"   New model output: {output_new}")
    print(f"   Difference: {output_new - output_old}")
    
    # Check if outputs are identical (or very close due to numerical precision)
    max_diff = torch.abs(output_new - output_old).max().item()
    mean_diff = torch.abs(output_new - output_old).mean().item()
    
    print(f"\n   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    
    # Tolerance check
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"\n   ✅ SUCCESS! Outputs are identical (within {tolerance} tolerance)")
        print("   The remapped checkpoint works exactly the same as the original!")
        return True
    else:
        print(f"\n   ⚠️  WARNING! Outputs differ by more than {tolerance}")
        print("   The mapping may need adjustment.")
        
        # Additional diagnostics
        print("\n6. Detailed layer-by-layer comparison...")
        
        # Compare feature extractor outputs
        with torch.no_grad():
            features_old = model_old.feature_extractor(test_input)
            features_new = model_new.mednextv1(test_input)
        
        feature_diff = torch.abs(features_new - features_old).max().item()
        print(f"   Feature extractor max diff: {feature_diff:.2e}")
        
        # Compare after pooling
        with torch.no_grad():
            pooled_old = model_old.global_avg_pool(features_old)
            pooled_new = model_new.global_avg_pool(features_new)
        
        pooled_diff = torch.abs(pooled_new - pooled_old).max().item()
        print(f"   After pooling max diff: {pooled_diff:.2e}")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

