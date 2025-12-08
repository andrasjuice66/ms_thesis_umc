#!/usr/bin/env python3
"""
Script to compare checkpoint architecture vs current model architecture.
Shows the key mapping needed to load old checkpoints into new models.
"""
import torch
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.models.brainagenext import BrainAgeNeXt


def print_section(title, char="="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}\n")


def organize_keys(keys):
    """Organize keys by their top-level prefix."""
    organized = defaultdict(list)
    for key in sorted(keys):
        parts = key.split('.')
        prefix = parts[0] if parts else key
        organized[prefix].append(key)
    return organized


def print_key_structure(organized_keys, max_keys_per_group=None):
    """Print organized key structure."""
    for prefix, keys in sorted(organized_keys.items()):
        print(f"\n  [{prefix}] ({len(keys)} keys)")
        for key in keys:
            print(f"    - {key}")


def map_checkpoint_keys(checkpoint_keys):
    """Map old checkpoint keys to new model keys."""
    mapping = {}
    for old_key in checkpoint_keys:
        new_key = old_key
        # Map feature_extractor.* to mednextv1.*
        if old_key.startswith('feature_extractor.'):
            new_key = old_key.replace('feature_extractor.', 'mednextv1.', 1)
        # Map regression_head.* to regression_fc.* with index adjustment
        # Old had nn.Flatten() as layer 0, so Linear layers were at indices 1 and 4
        # New removed nn.Flatten() from Sequential, so Linear layers are at indices 0 and 3
        elif old_key.startswith('regression_head.'):
            new_key = old_key.replace('regression_head.', 'regression_fc.', 1)
            # Fix index shift: 1→0, 4→3
            new_key = new_key.replace('regression_fc.1.', 'regression_fc.0.')
            new_key = new_key.replace('regression_fc.4.', 'regression_fc.3.')
        mapping[old_key] = new_key
    return mapping


def apply_mapping_and_save(checkpoint_path, output_path=None):
    """
    Apply the key mapping to checkpoint and save the remapped version.
    This creates a new checkpoint file that can be loaded directly into the current model.
    """
    print_section("APPLYING MAPPING AND SAVING REMAPPED CHECKPOINT", "=")
    
    # Load checkpoint
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
    
    # Get mapping
    mapping = map_checkpoint_keys(list(clean_state_dict.keys()))
    
    # Apply mapping
    new_state_dict = {}
    for old_key, new_key in mapping.items():
        if old_key in clean_state_dict:
            new_state_dict[new_key] = clean_state_dict[old_key]
    
    print(f"  Remapped {len(new_state_dict)} keys")
    
    # Create new checkpoint with same metadata
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        new_checkpoint = checkpoint.copy()
        new_checkpoint['state_dict'] = new_state_dict
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        new_checkpoint = checkpoint.copy()
        new_checkpoint['model_state_dict'] = new_state_dict
    else:
        new_checkpoint = new_state_dict
    
    # Save if output path provided
    if output_path:
        torch.save(new_checkpoint, output_path)
        print(f"  Saved remapped checkpoint to: {output_path}")
    
    return new_state_dict


def main():
    checkpoint_path = "/mnt/c/Projects/thesis_project/checkpoints_best_mae/brainagenext_seg_map/brainagenext_seg_map_best_mae.pt"
    
    print_section("CHECKPOINT vs CURRENT MODEL ARCHITECTURE COMPARISON")
    
    # 1. Load checkpoint
    print_section("1. CHECKPOINT STRUCTURE (Old Architecture)", "-")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                checkpoint_state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                checkpoint_state_dict = checkpoint['model_state_dict']
            else:
                checkpoint_state_dict = checkpoint
        else:
            checkpoint_state_dict = checkpoint
        
        # Remove 'module.' prefix if exists
        checkpoint_keys = []
        for k in checkpoint_state_dict.keys():
            clean_key = k[7:] if k.startswith('module.') else k
            checkpoint_keys.append(clean_key)
        
        checkpoint_organized = organize_keys(checkpoint_keys)
        print(f"Total keys in checkpoint: {len(checkpoint_keys)}")
        print_key_structure(checkpoint_organized)
        
        # Show checkpoint info if available
        if isinstance(checkpoint, dict):
            print(f"\n  Checkpoint metadata:")
            for key in ['epoch', 'best_metric', 'history']:
                if key in checkpoint:
                    print(f"    - {key}: {checkpoint[key]}")
    
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please update the checkpoint_path variable in the script.")
        return
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return
    
    # 2. Create current model
    print_section("2. CURRENT MODEL STRUCTURE (New Architecture)", "-")
    try:
        # Use same parameters as in config
        model = BrainAgeNeXt(
            in_channels=15,
            dropout_rate=0.2,
            model_id='B',
            kernel_size=3,
            deep_supervision=True,
            feature_size=512,
            hidden_size=64
        )
        
        model_keys = list(model.state_dict().keys())
        model_organized = organize_keys(model_keys)
        print(f"Total keys in current model: {len(model_keys)}")
        print_key_structure(model_organized)
        
    except Exception as e:
        print(f"ERROR creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Show key mapping
    print_section("3. KEY MAPPING (Old → New)", "-")
    mapping = map_checkpoint_keys(checkpoint_keys)
    
    # Group mappings by type
    feature_extractor_mappings = {k: v for k, v in mapping.items() if k.startswith('feature_extractor.')}
    regression_head_mappings = {k: v for k, v in mapping.items() if k.startswith('regression_head.')}
    other_mappings = {k: v for k, v in mapping.items() if not k.startswith(('feature_extractor.', 'regression_head.'))}
    
    print(f"\n  Feature Extractor mappings (feature_extractor.* → mednextv1.*):")
    print(f"    Total: {len(feature_extractor_mappings)}")
    for i, (old, new) in enumerate(list(feature_extractor_mappings.items())):
        print(f"    {old}")
        print(f"      → {new}")
    
    print(f"\n  Regression Head mappings (regression_head.* → regression_fc.*):")
    print(f"    Total: {len(regression_head_mappings)}")
    for old, new in regression_head_mappings.items():
        print(f"    {old}")
        print(f"      → {new}")
    
    if other_mappings:
        print(f"\n  Other keys (no mapping needed):")
        print(f"    Total: {len(other_mappings)}")
        for old, new in list(other_mappings.items()):
            print(f"    {old}")
    
    # 4. Check compatibility
    print_section("4. COMPATIBILITY ANALYSIS", "-")
    
    # Apply mapping to checkpoint keys
    mapped_checkpoint_keys = set(mapping.values())
    model_keys_set = set(model_keys)
    
    matching_keys = mapped_checkpoint_keys & model_keys_set
    missing_in_model = mapped_checkpoint_keys - model_keys_set
    missing_in_checkpoint = model_keys_set - mapped_checkpoint_keys
    
    print(f"\n  Keys that will match after mapping: {len(matching_keys)}")
    for key in sorted(matching_keys):
        print(f"    ✓ {key}")
    
    print(f"\n  Keys in checkpoint (after mapping) but NOT in model: {len(missing_in_model)}")
    if missing_in_model:
        for key in sorted(list(missing_in_model)):
            print(f"    ✗ {key}")
    
    print(f"\n  Keys in model but NOT in checkpoint: {len(missing_in_checkpoint)}")
    if missing_in_checkpoint:
        for key in sorted(list(missing_in_checkpoint)):
            print(f"    ? {key}")
    
    # 5. Shape comparison for matching keys
    print_section("5. SHAPE COMPARISON (Matching Keys)", "-")
    
    # Create a model state dict with mapped checkpoint keys
    mapped_checkpoint_state_dict = {}
    for old_key, new_key in mapping.items():
        if old_key in checkpoint_state_dict:
            mapped_checkpoint_state_dict[new_key] = checkpoint_state_dict[old_key]
    
    model_state_dict = model.state_dict()
    shape_matches = []
    shape_mismatches = []
    
    for key in matching_keys:
        if key in mapped_checkpoint_state_dict and key in model_state_dict:
            ckpt_shape = mapped_checkpoint_state_dict[key].shape
            model_shape = model_state_dict[key].shape
            if ckpt_shape == model_shape:
                shape_matches.append((key, ckpt_shape))
            else:
                shape_mismatches.append((key, ckpt_shape, model_shape))
    
    print(f"\n  Keys with matching shapes: {len(shape_matches)}")
    for key, shape in shape_matches:
        print(f"    ✓ {key}: {shape}")
    
    if shape_mismatches:
        print(f"\n  Keys with MISMATCHING shapes: {len(shape_mismatches)}")
        for key, ckpt_shape, model_shape in shape_mismatches:
            print(f"    ✗ {key}")
            print(f"      Checkpoint: {ckpt_shape}")
            print(f"      Model:      {model_shape}")
    
    # 6. Summary
    print_section("6. SUMMARY", "-")
    print(f"""
  Checkpoint keys:        {len(checkpoint_keys)}
  Model keys:             {len(model_keys)}
  Keys after mapping:     {len(mapped_checkpoint_keys)}
  Matching keys:          {len(matching_keys)}
  Shape matches:          {len(shape_matches)}
  Shape mismatches:       {len(shape_mismatches)}
  Missing in model:       {len(missing_in_model)}
  Missing in checkpoint: {len(missing_in_checkpoint)}
  
  Load success rate:      {len(shape_matches)}/{len(checkpoint_keys)} ({100*len(shape_matches)/len(checkpoint_keys):.1f}%)
    """)
    
    print_section("END OF COMPARISON", "=")
    
    # 7. Apply mapping and save remapped checkpoint
    print_section("7. CREATING REMAPPED CHECKPOINT", "-")
    
    output_path = checkpoint_path.replace('.pt', '_remapped.pt')
    remapped_state_dict = apply_mapping_and_save(checkpoint_path, output_path)
    
    # 8. Test loading remapped checkpoint into model
    print_section("8. TESTING REMAPPED CHECKPOINT", "-")
    try:
        result = model.load_state_dict(remapped_state_dict, strict=False)
        print(f"  ✓ Successfully loaded remapped checkpoint into model!")
        print(f"  Missing keys: {len(result.missing_keys)}")
        if result.missing_keys:
            for key in result.missing_keys:
                print(f"    - {key}")
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        if result.unexpected_keys:
            for key in result.unexpected_keys:
                print(f"    - {key}")
        
        if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
            print(f"\n  🎉 PERFECT MATCH! All keys loaded successfully!")
            print(f"  You can now use the remapped checkpoint: {output_path}")
    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
    
    print_section("COMPLETE", "=")


if __name__ == "__main__":
    main()