#!/usr/bin/env python3
"""
Compare the actual keys in both checkpoints' model_state_dict.
"""
import torch

old_path = "/mnt/c/Projects/thesis_project/checkpoints_best_mae/brainagenext_seg_map/brainagenext_seg_map_best_mae.pt"
remapped_path = "/mnt/c/Projects/thesis_project/checkpoints_best_mae/brainagenext_seg_map/brainagenext_seg_map_best_mae_remapped.pt"

print("="*80)
print("ORIGINAL CHECKPOINT - model_state_dict keys:")
print("="*80)
old_ckpt = torch.load(old_path, map_location='cpu')
old_state = old_ckpt['model_state_dict']
print(f"\nTotal keys: {len(old_state)}")
print("\nFirst 20 keys:")
for i, key in enumerate(list(old_state.keys())[:20]):
    print(f"  {i+1}. {key}")

print("\n" + "="*80)
print("REMAPPED CHECKPOINT - model_state_dict keys:")
print("="*80)
remapped_ckpt = torch.load(remapped_path, map_location='cpu')
remapped_state = remapped_ckpt['model_state_dict']
print(f"\nTotal keys: {len(remapped_state)}")
print("\nFirst 20 keys:")
for i, key in enumerate(list(remapped_state.keys())[:20]):
    print(f"  {i+1}. {key}")

print("\n" + "="*80)
print("COMPARISON:")
print("="*80)

old_keys = set(old_state.keys())
remapped_keys = set(remapped_state.keys())

only_in_old = old_keys - remapped_keys
only_in_remapped = remapped_keys - old_keys

if only_in_old:
    print(f"\nKeys only in ORIGINAL (not in remapped): {len(only_in_old)}")
    for key in sorted(list(only_in_old)[:10]):
        print(f"  - {key}")

if only_in_remapped:
    print(f"\nKeys only in REMAPPED (not in original): {len(only_in_remapped)}")
    for key in sorted(list(only_in_remapped)[:10]):
        print(f"  + {key}")

if not only_in_old and not only_in_remapped:
    print("\n✓ Both checkpoints have identical keys!")

