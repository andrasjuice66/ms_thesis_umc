#!/usr/bin/env python
"""
Quick test to validate brain generator works correctly.
"""
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.brain_gen.brain_generator import BABrainGenerator
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES, GENERATION_LABELS, N_NEUTRAL_LABELS

def quick_test():
    print("Quick Brain Generator Test")
    print("-" * 30)
    
    # Create minimal brain generator
    n_classes = GENERATION_CLASSES.max() + 1
    
    prior_means = np.vstack([
        np.full(n_classes, 125.0, dtype=float),
        np.full(n_classes, 125.0, dtype=float),
    ])
    
    prior_stds = np.vstack([
        np.full(n_classes, 17.5, dtype=float),
        np.full(n_classes, 17.5, dtype=float),
    ])
    
    prior_means[:, 0] = 0.0    
    prior_stds[:, 0] = 0.0     
    
    # Minimal probabilities (no augmentations)
    prob = {k: 0.0 for k in ["flip", "affine", "contrast", "gamma", "scale_int", 
                             "shift_int", "hist_shift", "noise", "rician", "gibbs", 
                             "blur", "bias", "resolution"]}
    
    brain_gen = BABrainGenerator(
        prior_means=prior_means,
        prior_stds=prior_stds,
        distribution="normal",
        prob=prob,
        rotation_range=0,
        scaling_range=0,
        shear_bounds=0,
        translation_bounds=0,
        contrast_range=(1.0, 1.0),
        log_gamma_std=0,
        shift_offset=0,
        hist_control_points=5,
        noise_mean=0,
        noise_std=0,
        rician_std=0,
        gibbs_alpha=0,
        blur_sigma=0,
        bias_field_rng=(0.0, 0.0),
        min_res=1.0,
        max_res_iso=1.0,
        max_res_aniso=1.0,
        atlas_res=1.0,
        generation_labels=GENERATION_LABELS,
        n_neutral_labels=N_NEUTRAL_LABELS,
        use_hemisphere_aware_flip=False,
        use_dynamic_resolution=False,
        use_intensity_clip_normalize=False,
        n_channels=1,
        output_shape=(64, 64, 64),  # Smaller for quick test
        use_random_cropping=False,
    )
    
    print("✓ Brain generator created")
    
    # Create dummy segmentation
    seg_data = np.random.randint(0, 15, size=(1, 64, 64, 64)).astype(np.float32)
    seg_tensor = torch.from_numpy(seg_data)
    
    print(f"✓ Test segmentation created: {seg_tensor.shape}")
    
    # Test generation
    try:
        result = brain_gen({"image": seg_tensor})
        generated = result["image"]
        
        print(f"✓ Generation successful!")
        print(f"  Input shape: {seg_tensor.shape}")
        print(f"  Output shape: {generated.shape}")
        print(f"  Output range: [{generated.min():.3f}, {generated.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1) 