#!/usr/bin/env python3
"""
Inspect SynthSeg .h5 model file structure.
Run this before attempting weight transfer to understand the model architecture.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse


def inspect_h5_structure(h5_path: str, max_depth: int = 3, show_weights: bool = False):
    """
    Inspect the structure of an HDF5 file.
    
    Args:
        h5_path: Path to .h5 file
        max_depth: Maximum depth to traverse
        show_weights: Whether to show weight shapes and statistics
    """
    print(f"🔍 Inspecting: {h5_path}")
    print("=" * 80)
    
    def print_structure(name, obj, depth=0):
        if depth > max_depth:
            return
            
        indent = "  " * depth
        
        if isinstance(obj, h5py.Group):
            print(f"{indent}📁 {name}/ (Group)")
            # Recurse into group
            for key in obj.keys():
                print_structure(f"{name}/{key}" if name else key, obj[key], depth + 1)
                
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            size_mb = obj.size * obj.dtype.itemsize / 1024 / 1024
            
            print(f"{indent}📊 {name} (Dataset)")
            print(f"{indent}   Shape: {shape}")
            print(f"{indent}   Dtype: {dtype}")
            print(f"{indent}   Size: {size_mb:.2f} MB")
            
            if show_weights and obj.size > 0:
                try:
                    data = obj[:]
                    if data.size > 0:
                        print(f"{indent}   Min: {np.min(data):.6f}")
                        print(f"{indent}   Max: {np.max(data):.6f}")
                        print(f"{indent}   Mean: {np.mean(data):.6f}")
                        print(f"{indent}   Std: {np.std(data):.6f}")
                except Exception as e:
                    print(f"{indent}   ⚠️  Could not read data: {e}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"📋 File info:")
            print(f"   Keys at root: {list(f.keys())}")
            print(f"   File size: {Path(h5_path).stat().st_size / 1024 / 1024:.2f} MB")
            print()
            
            # Show structure
            print("🏗️  Structure:")
            for key in f.keys():
                print_structure(key, f[key])
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def find_layer_patterns(h5_path: str):
    """Find common layer naming patterns in the .h5 file."""
    print("\n🎯 Layer Naming Patterns:")
    print("=" * 50)
    
    patterns = {
        'encoder': [],
        'decoder': [],
        'normalization': [],
        'activation': [],
        'other': []
    }
    
    try:
        with h5py.File(h5_path, 'r') as f:
            def collect_names(name, obj):
                if isinstance(obj, h5py.Dataset):
                    name_lower = name.lower()
                    
                    if 'downarm' in name_lower or 'encoder' in name_lower:
                        patterns['encoder'].append(name)
                    elif 'uparm' in name_lower or 'decoder' in name_lower:
                        patterns['decoder'].append(name)
                    elif any(norm in name_lower for norm in ['batch_norm', 'instance_norm', 'bn']):
                        patterns['normalization'].append(name)
                    elif any(act in name_lower for act in ['relu', 'elu', 'activation']):
                        patterns['activation'].append(name)
                    else:
                        patterns['other'].append(name)
            
            f.visititems(collect_names)
            
        for category, names in patterns.items():
            if names:
                print(f"\n{category.upper()} ({len(names)} layers):")
                for name in sorted(names[:5]):  # Show first 5
                    print(f"  {name}")
                if len(names) > 5:
                    print(f"  ... and {len(names) - 5} more")
                    
    except Exception as e:
        print(f"❌ Error analyzing patterns: {e}")


def compare_with_pytorch_model():
    """Show what your PyTorch model structure looks like for comparison."""
    print("\n🏗️  Your PyTorch Model Structure:")
    print("=" * 50)
    
    try:
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        
        from brain_age_pred.models.multi_head import MultiTaskBrainAge
        from brain_age_pred.brain_gen.labels import GENERATION_CLASSES
        
        model = MultiTaskBrainAge(n_classes=GENERATION_CLASSES.max() + 1)
        
        print("PyTorch model layers:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")
            
    except Exception as e:
        print(f"❌ Could not load PyTorch model: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect SynthSeg .h5 model file")
    parser.add_argument("h5_path", help="Path to SynthSeg .h5 file")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth to traverse")
    parser.add_argument("--show-weights", action="store_true", help="Show weight statistics")
    parser.add_argument("--compare-pytorch", action="store_true", help="Compare with PyTorch model")
    
    args = parser.parse_args()
    
    if not Path(args.h5_path).exists():
        print(f"❌ File not found: {args.h5_path}")
        return
    
    # Main inspection
    inspect_h5_structure(args.h5_path, args.max_depth, args.show_weights)
    
    # Find patterns
    find_layer_patterns(args.h5_path)
    
    # Compare with PyTorch model if requested
    if args.compare_pytorch:
        compare_with_pytorch_model()
    
    print("\n✅ Inspection complete!")
    print("Use this information to adjust the weight mapping in weight_transfer.py")


if __name__ == "__main__":
    main() 