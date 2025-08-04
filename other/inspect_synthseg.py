import h5py
import numpy as np

def inspect_synthseg_model(h5_path: str):
    """Inspect the structure of a SynthSeg .h5 model"""
    print(f"Inspecting SynthSeg model: {h5_path}")
    print("=" * 60)
    
    with h5py.File(h5_path, 'r') as f:
        def print_structure(name, obj, level=0):
            indent = "  " * level
            if isinstance(obj, h5py.Group):
                print(f"{indent}{name}/")
            elif isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                print(f"{indent}{name}: {shape} ({dtype})")
        
        print("HDF5 Structure:")
        f.visititems(print_structure)
        
        print("\n" + "=" * 60)
        print("Weights Summary:")
        
        weights = {}
        def collect_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)
        
        f.visititems(collect_weights)
        
        # Group by layer type
        conv_weights = [k for k in weights.keys() if 'conv' in k.lower() and ('kernel' in k or 'weight' in k)]
        norm_weights = [k for k in weights.keys() if ('batch' in k.lower() or 'instance' in k.lower()) and ('gamma' in k or 'beta' in k or 'weight' in k or 'bias' in k)]
        other_weights = [k for k in weights.keys() if k not in conv_weights and k not in norm_weights]
        
        print(f"\nConvolution layers ({len(conv_weights)}):")
        for key in sorted(conv_weights):
            print(f"  {key}: {weights[key].shape}")
        
        print(f"\nNormalization layers ({len(norm_weights)}):")
        for key in sorted(norm_weights):
            print(f"  {key}: {weights[key].shape}")
        
        print(f"\nOther layers ({len(other_weights)}):")
        for key in sorted(other_weights):
            print(f"  {key}: {weights[key].shape}")

# Usage:
if __name__ == "__main__":
    synthseg_path = "/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/OtherRepos/SynthSeg/models/synthseg_2.0.h5"
    inspect_synthseg_model(synthseg_path)