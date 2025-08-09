#!/usr/bin/env python
"""
Script to inspect the contents of PyTorch checkpoint files.
This helps debug parameter loading issues by showing what keys are available.
"""

import sys
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.models.sfcn_class import SFCNClass

def inspect_checkpoint(checkpoint_path):
    """Inspect the contents of a checkpoint file."""
    print(f"\n{'='*80}")
    print(f"INSPECTING CHECKPOINT: {checkpoint_path}")
    print(f"{'='*80}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📁 Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Find the state dict
            state_dict = None
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"🔍 Using 'state_dict' key")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"🔍 Using 'model_state_dict' key")
            else:
                state_dict = checkpoint
                print(f"🔍 Using checkpoint as state_dict")
                
            # Display additional info
            if 'epoch' in checkpoint:
                print(f"📅 Epoch: {checkpoint['epoch']}")
            if 'best_metric' in checkpoint:
                print(f"🏆 Best metric: {checkpoint['best_metric']}")
                
        else:
            state_dict = checkpoint
            print(f"🔍 Checkpoint is a direct state_dict")
        
        # Remove 'module.' prefix if present
        clean_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            clean_state_dict[name] = v
            
        print(f"\n📊 PARAMETER ANALYSIS")
        print(f"{'─'*80}")
        print(f"Total parameters in checkpoint: {len(clean_state_dict)}")
        
        # Categorize parameters
        conv_params = []
        norm_params = []
        classifier_params = []
        running_stats = []
        
        for key, tensor in clean_state_dict.items():
            if 'running_mean' in key or 'running_var' in key:
                running_stats.append((key, tensor.shape))
            elif 'feature_extractor' in key and ('weight' in key or 'bias' in key):
                if 'conv' in key.lower() or '0.weight' in key or '0.bias' in key:
                    conv_params.append((key, tensor.shape))
                elif '1.weight' in key or '1.bias' in key:
                    norm_params.append((key, tensor.shape))
            elif 'classifier' in key:
                classifier_params.append((key, tensor.shape))
            else:
                print(f"❓ Unclassified: {key}: {tensor.shape}")
        
        print(f"\n🔸 Convolutional layers: {len(conv_params)}")
        for key, shape in conv_params:
            print(f"   {key}: {shape}")
            
        print(f"\n🔸 Normalization layers: {len(norm_params)}")
        for key, shape in norm_params:
            print(f"   {key}: {shape}")
            
        print(f"\n🔸 Running statistics: {len(running_stats)}")
        for key, shape in running_stats:
            print(f"   {key}: {shape}")
            
        print(f"\n🔸 Classifier layers: {len(classifier_params)}")
        for key, shape in classifier_params:
            print(f"   {key}: {shape}")
            
        # Create model for comparison
        print(f"\n🔄 MODEL COMPARISON")
        print(f"{'─'*80}")
        
        # Test with both track_running_stats settings
        for track_stats in [False, True]:
            print(f"\n🧪 Testing with track_running_stats={track_stats}")
            model = SFCNClass(
                in_channels=1,
                dropout_rate=0.3,
                age_min=20,
                age_max=80,
                track_running_stats=track_stats
            )
            
            model_state_dict = model.state_dict()
            print(f"   Model expects: {len(model_state_dict)} parameters")
            
            # Check compatibility
            compatible = 0
            incompatible = []
            
            for key in clean_state_dict:
                if key in model_state_dict:
                    if clean_state_dict[key].shape == model_state_dict[key].shape:
                        compatible += 1
                    else:
                        incompatible.append(f"{key}: shape mismatch")
                else:
                    incompatible.append(f"{key}: missing in model")
            
            print(f"   Compatible parameters: {compatible}/{len(clean_state_dict)}")
            if incompatible:
                print(f"   ⚠️  Incompatible:")
                for issue in incompatible[:5]:  # Show first 5
                    print(f"      {issue}")
                if len(incompatible) > 5:
                    print(f"      ... and {len(incompatible) - 5} more")
                    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

def main():
    checkpoint_path = "/mnt/c/Projects/thesis_project/checkpoints/sfcn_class_seg_map/sfcn_class_seg_map_best_mae.pt"

    inspect_checkpoint(checkpoint_path)
    
    print(f"\n{'='*80}")
    print("INSPECTION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 