#!/usr/bin/env python3
"""
Inspect the structure of both checkpoint files to debug the issue.
"""
import torch

def inspect_checkpoint(path, name):
    print(f"\n{'='*80}")
    print(f"Inspecting: {name}")
    print(f"Path: {path}")
    print(f"{'='*80}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    print(f"\nType: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nTop-level keys: {list(checkpoint.keys())}")
        
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"\n  '{key}' is a dict with {len(value)} items")
                if len(value) < 20:
                    print(f"    Keys: {list(value.keys())[:10]}")
            elif isinstance(value, torch.Tensor):
                print(f"\n  '{key}' is a Tensor with shape {value.shape}")
            else:
                print(f"\n  '{key}' = {value}")
        
        # Check state_dict specifically
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n{'='*80}")
            print(f"state_dict contents:")
            print(f"  Total keys: {len(state_dict)}")
            print(f"  First 10 keys:")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"    {i+1}. {key}: {state_dict[key].shape}")
    else:
        print(f"\nNot a dict - contains {len(checkpoint)} items")
        print(f"First 10 keys:")
        for i, key in enumerate(list(checkpoint.keys())[:10]):
            print(f"  {i+1}. {key}: {checkpoint[key].shape}")


if __name__ == "__main__":
    old_path = "/mnt/c/Projects/thesis_project/checkpoints_best_mae/brainagenext_seg_map/brainagenext_seg_map_best_mae.pt"
    remapped_path = "/mnt/c/Projects/thesis_project/checkpoints_best_mae/brainagenext_seg_map/brainagenext_seg_map_best_mae_remapped.pt"
    
    inspect_checkpoint(old_path, "ORIGINAL CHECKPOINT")
    inspect_checkpoint(remapped_path, "REMAPPED CHECKPOINT")

