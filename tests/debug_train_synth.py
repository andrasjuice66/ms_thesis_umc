#!/usr/bin/env python
"""
Debug version of train_synth.py that saves intermediate outputs
and validates brain generator at each step.
"""
import os, sys, time, json, random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dataset.dataset import BADataset          
from brain_age_pred.brain_gen.brain_generator import BABrainGenerator
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES, GENERATION_LABELS, N_NEUTRAL_LABELS
from brain_age_pred.utils.utils import read_csv

def debug_brain_generator(cfg, output_dir):
    """Debug brain generator setup and save samples"""
    print("=== DEBUG: Brain Generator Setup ===")
    
    # Setup brain generator (copy from train_synth.py)
    bg_cfg = cfg.get("brain_generator", {})
    
    mean_loc = bg_cfg.get("mean_loc", 125.0)
    mean_scale = bg_cfg.get("mean_scale", 125.0)
    std_loc = bg_cfg.get("std_loc", 17.5)
    std_scale = bg_cfg.get("std_scale", 17.5)
    
    prob = bg_cfg.get("prob", {
        "flip": 0.5,
        "affine": 0.0,
        "contrast": 0.3,
        "gamma": 0.3,
        "scale_int": 0.3,
        "shift_int": 0.3,
        "hist_shift": 0.3,
        "noise": 0.3,
        "rician": 0.3,
        "gibbs": 0.3,
        "blur": 0.3,
        "bias": 0.0,
        "resolution": 0.3,
    })
    
    n_classes = GENERATION_CLASSES.max() + 1
    
    prior_means = np.vstack([
        np.full(n_classes, mean_loc, dtype=float),
        np.full(n_classes, mean_scale, dtype=float),
    ])
    
    prior_stds = np.vstack([
        np.full(n_classes, std_loc, dtype=float),
        np.full(n_classes, std_scale, dtype=float),
    ])
    
    prior_means[:, 0] = 0.0    
    prior_stds[:, 0] = 0.0     
    
    brain_generator = BABrainGenerator(
        prior_means=prior_means,
        prior_stds=prior_stds,
        distribution=bg_cfg.get("distribution", "normal"),
        prob=prob,
        rotation_range=bg_cfg.get("rotation_range", 10),
        scaling_range=bg_cfg.get("scaling_range", 0.1),
        shear_bounds=bg_cfg.get("shear_bounds", 0.005),
        translation_bounds=bg_cfg.get("translation_bounds", False),
        contrast_range=tuple(bg_cfg.get("contrast_range", [0.8, 1.2])),
        log_gamma_std=bg_cfg.get("log_gamma_std", 0.1),
        shift_offset=bg_cfg.get("shift_offset", 0.1),
        hist_control_points=bg_cfg.get("hist_control_points", 5),
        noise_mean=bg_cfg.get("noise_mean", 0.02),
        noise_std=bg_cfg.get("noise_std", 0.015),
        rician_std=bg_cfg.get("rician_std", 0.01),
        gibbs_alpha=bg_cfg.get("gibbs_alpha", 0.4),
        blur_sigma=bg_cfg.get("blur_sigma", 0.25),
        bias_field_rng=tuple(bg_cfg.get("bias_field_rng", [0.0, 0.5])),
        min_res=bg_cfg.get("min_res", 0.8),
        max_res_iso=bg_cfg.get("max_res_iso", 2.0),
        max_res_aniso=bg_cfg.get("max_res_aniso", 2.0),
        atlas_res=bg_cfg.get("atlas_res", 1.0),
        thickness=bg_cfg.get("thickness", None),
        generation_labels=GENERATION_LABELS,
        n_neutral_labels=N_NEUTRAL_LABELS,
        output_labels=None,
        use_hemisphere_aware_flip=bg_cfg.get("use_hemisphere_aware_flip", True),
        use_dynamic_resolution=bg_cfg.get("use_dynamic_resolution", True),
        use_intensity_clip_normalize=bg_cfg.get("use_intensity_clip_normalize", True),
        n_channels=bg_cfg.get("n_channels", 1),
        use_specific_stats_for_channel=bg_cfg.get("use_specific_stats_for_channel", False),
        output_shape=tuple(bg_cfg.get("output_shape", [182, 218, 182])),
        use_random_cropping=bg_cfg.get("use_random_cropping", True),
        return_gradients=bg_cfg.get("return_gradients", False),
    )
    
    print("✓ Brain generator created successfully")
    print(f"  Prior means shape: {prior_means.shape}")
    print(f"  Prior stds shape: {prior_stds.shape}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Output shape: {bg_cfg.get('output_shape', [182, 218, 182])}")
    
    return brain_generator

def debug_dataset_loading(cfg, brain_generator, output_dir, max_samples=5):
    """Debug dataset loading and save intermediate results"""
    print("\n=== DEBUG: Dataset Loading ===")
    
    # Read training data (same as train_synth.py)
    train_csv = Path(cfg.get("data.train_csv"))
    segmented_data_dir = Path(cfg.get("data.segmented_data_dir"))
    
    print(f"Reading train CSV: {train_csv}")
    print(f"Segmented data dir: {segmented_data_dir}")
    
    train_p, train_a, train_w, train_s, train_m = read_csv(train_csv, segmented_data_dir)
    
    print(f"✓ Read {len(train_p)} training samples")
    print(f"  Sample file extensions: {set(Path(p).suffix for p in train_p[:10])}")
    print(f"  Sample paths: {train_p[:3]}")
    print(f"  Sample ages: {train_a[:5]}")
    
    # Create dataset
    dataset = BADataset(
        file_paths=train_p[:max_samples],  # Limit for debugging
        age_labels=train_a[:max_samples],
        sample_wts=train_w[:max_samples] if train_w else None,
        sexes=train_s[:max_samples] if train_s else None,
        modalities=train_m[:max_samples] if train_m else None,
        transform=brain_generator,
        mode="train",
        cache_size=0,
    )
    
    print(f"✓ Created dataset with {len(dataset)} samples")
    
    # Test each sample
    debug_samples_dir = output_dir / "debug_samples"
    debug_samples_dir.mkdir(exist_ok=True)
    
    for i in range(len(dataset)):
        print(f"\n--- Processing sample {i} ---")
        try:
            # Load raw file first
            raw_data = dataset._load_volume(dataset.file_paths[i])
            print(f"  Raw data shape: {raw_data.shape}")
            print(f"  Raw data type: {raw_data.dtype}")
            print(f"  Raw data range: [{raw_data.min():.3f}, {raw_data.max():.3f}]")
            print(f"  Unique raw values: {len(np.unique(raw_data))}")
            
            # Save raw data
            np.save(debug_samples_dir / f"sample_{i}_raw.npy", raw_data)
            
            # Process through dataset (includes brain generator)
            sample = dataset[i]
            generated_img = sample['image'].cpu().numpy()
            
            print(f"  Generated shape: {generated_img.shape}")
            print(f"  Generated type: {generated_img.dtype}")
            print(f"  Generated range: [{generated_img.min():.3f}, {generated_img.max():.3f}]")
            print(f"  Age: {sample['age'].item():.1f}")
            
            # Save generated data
            np.save(debug_samples_dir / f"sample_{i}_generated.npy", generated_img)
            
            # Save metadata
            metadata = {
                "file_path": dataset.file_paths[i],
                "age": sample['age'].item(),
                "raw_shape": raw_data.shape,
                "raw_range": [float(raw_data.min()), float(raw_data.max())],
                "generated_shape": generated_img.shape,
                "generated_range": [float(generated_img.min()), float(generated_img.max())],
            }
            
            with open(debug_samples_dir / f"sample_{i}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Saved to debug_samples/sample_{i}_*")
            
        except Exception as e:
            print(f"  ✗ Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    return dataset

def debug_dataloader(dataset, output_dir, batch_size=2):
    """Debug dataloader behavior"""
    print(f"\n=== DEBUG: DataLoader (batch_size={batch_size}) ===")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for debugging
        num_workers=0,  # Single process for debugging
    )
    
    debug_batches_dir = output_dir / "debug_batches"
    debug_batches_dir.mkdir(exist_ok=True)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:  # Only test first 2 batches
            break
            
        print(f"\n--- Batch {batch_idx} ---")
        print(f"  Batch image shape: {batch['image'].shape}")
        print(f"  Batch image dtype: {batch['image'].dtype}")
        print(f"  Batch ages: {batch['age'].tolist()}")
        print(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        
        # Save batch data
        for sample_idx in range(batch['image'].shape[0]):
            img = batch['image'][sample_idx].cpu().numpy()
            age = batch['age'][sample_idx].item()
            
            np.save(debug_batches_dir / f"batch_{batch_idx}_sample_{sample_idx}.npy", img)
            
            metadata = {
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "age": age,
                "shape": img.shape,
                "range": [float(img.min()), float(img.max())],
            }
            
            with open(debug_batches_dir / f"batch_{batch_idx}_sample_{sample_idx}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved batch {batch_idx} to debug_batches/")
    
    print("✓ DataLoader debugging complete")

def main():
    """Main debug function"""
    print("DEBUG BRAIN GENERATOR TRAINING PIPELINE")
    print("=" * 60)
    
    # Setup
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "brain_age_pred/configs/segmented/brainagenext_segmented_snell_debug.yaml"
    cfg = Config(cfg_file)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path("debug_output") / f"brain_gen_debug_{timestamp}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Config file: {cfg_file}")
    print(f"Debug output: {debug_dir}")
    
    try:
        # Debug brain generator
        brain_generator = debug_brain_generator(cfg, debug_dir)
        
        # Debug dataset loading
        dataset = debug_dataset_loading(cfg, brain_generator, debug_dir, max_samples=5)
        
        # Debug dataloader
        debug_dataloader(dataset, debug_dir)
        
        print(f"\n{'='*60}")
        print("✓ DEBUG COMPLETE - All components working!")
        print(f"✓ Results saved to: {debug_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ DEBUG FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 