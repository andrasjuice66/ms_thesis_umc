#!/usr/bin/env python
"""
Data Pipeline Visualization Script
Shows exactly what data is fed into the model by sampling from the dataset
"""
import os, sys, time, json, random
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dataset.dataset import BADataset
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv
from brain_age_pred.brain_gen.brain_generator import BABrainGenerator
from brain_age_pred.brain_gen.labels import GENERATION_LABELS, GENERATION_CLASSES, N_NEUTRAL_LABELS


def save_image_as_nifti(image_tensor, save_path, affine=None):
    """Save a torch tensor as NIfTI file using nibabel."""
    # Convert tensor to numpy and remove channel dimension if present
    if len(image_tensor.shape) == 4:  # (C, D, H, W)
        image_np = image_tensor.squeeze(0).cpu().numpy()
    else:  # (D, H, W)
        image_np = image_tensor.cpu().numpy()
    
    # Use identity matrix if no affine provided
    if affine is None:
        affine = np.eye(4)
    
    # Create NIfTI image using nibabel and save
    nifti_img = nib.Nifti1Image(image_np, affine)
    nib.save(nifti_img, str(save_path))


def save_image_as_numpy(image_tensor, save_path):
    """Save a torch tensor as numpy array (.npy file)."""
    # Convert tensor to numpy and remove channel dimension if present
    if len(image_tensor.shape) == 4:  # (C, D, H, W)
        image_np = image_tensor.squeeze(0).cpu().numpy()
    else:  # (D, H, W)
        image_np = image_tensor.cpu().numpy()
    
    # Save as numpy array
    np.save(str(save_path), image_np)


def save_dataset_batch(dataloader, output_dir, prefix, num_batches=5, save_format="nifti"):
    """Save batches from dataloader to visualize what's fed to the model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Saving {num_batches} {prefix} batches to {output_dir} (format: {save_format}) ===")
    
    metadata = []
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        # Create batch directory
        batch_dir = output_dir / f"batch_{batch_idx}"
        batch_dir.mkdir(exist_ok=True)
        
        # Extract batch data
        images = batch["image"]
        ages = batch["age"]
        paths = batch.get("__image_path__", [f"unknown_{i}" for i in range(len(images))])
        
        print(f"  Saving batch {batch_idx+1}/{num_batches} with {len(images)} samples...")
        
        # Process each sample in the batch
        for i, (image, age, path) in enumerate(zip(images, ages, paths)):
            try:
                # Create filename with metadata
                original_name = Path(path).stem if isinstance(path, str) else f"sample_{i}"
                
                if save_format.lower() == "nifti":
                    filename = f"{prefix}_batch{batch_idx}_sample{i}_{original_name}_age{age.item():.1f}.nii.gz"
                    save_path = batch_dir / filename
                    save_image_as_nifti(image, save_path)
                elif save_format.lower() == "numpy":
                    filename = f"{prefix}_batch{batch_idx}_sample{i}_{original_name}_age{age.item():.1f}.npy"
                    save_path = batch_dir / filename
                    save_image_as_numpy(image, save_path)
                else:
                    raise ValueError(f"Unsupported save format: {save_format}")
                
                # Store metadata
                metadata.append({
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "filename": str(save_path.relative_to(output_dir)),
                    "age": age.item(),
                    "original_path": str(path) if isinstance(path, str) else f"unknown_{i}",
                    "image_shape": list(image.shape),
                    "image_min": float(image.min().item()),
                    "image_max": float(image.max().item()),
                    "image_mean": float(image.mean().item()),
                    "image_std": float(image.std().item()),
                })
                
            except Exception as e:
                print(f"  Error saving sample {i} in batch {batch_idx}: {e}")
                continue
    
    # Save metadata as JSON
    metadata_path = output_dir / f"{prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Completed! Saved {len(metadata)} samples from {min(num_batches, batch_idx+1)} batches")
    return metadata


def main():
    # 1. ─── configuration & reproducibility ─────────────────── #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic/brainagenext_synthetic.yaml"
    cfg = Config(cfg_file)
    
    # 2. ─── experiment naming / I/O ─────────────────────────── #
    experiment_name = cfg.get("output.experiment_name")
    if not experiment_name:
        experiment_name = f'model_input_debug_{timestamp}'
    
    out_root = Path("debug_output")
    output_dir = out_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("model-input-debug", log_file=output_dir / "debug.log")
    
    logger.info("Initializing model input debugging...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Experiment: {experiment_name}\nConfig: {cfg_file}")
    logger.info(f"Output directory: {output_dir}")

    # 3. ─── Set up device ─────────────────────────────────────── #
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 4. ─── Initialize Brain Generator for synthetic data ────── #
    logger.info("Initializing Brain Generator...")
    bg_cfg = cfg.get("brain_generator", {})
    
    # Augmentation probabilities from config
    prob = bg_cfg.get("prob", {})
    
    # Prior distribution parameters
    mean_loc = bg_cfg.get("mean_loc", 125.0)
    mean_scale = bg_cfg.get("mean_scale", 100.0)
    std_loc = bg_cfg.get("std_loc", 15.0)
    std_scale = bg_cfg.get("std_scale", 10.0)
    
    n_classes = len(GENERATION_LABELS)

    # "loc" = mid-point, "scale" = half-range
    prior_means = np.vstack([
        np.full(n_classes, mean_loc, dtype=float),
        np.full(n_classes, mean_scale, dtype=float),
    ])

    prior_stds = np.vstack([
        np.full(n_classes, std_loc, dtype=float),
        np.full(n_classes, std_scale, dtype=float),
    ])

    # Set background class (label 0) to zero
    prior_means[:, 0] = 0.0    
    prior_stds[:, 0] = 0.0

    # Initialize brain generator for synthetic data
    brain_generator = BABrainGenerator(
        # Required parameters
        prior_means=prior_means,
        prior_stds=prior_stds,
        distribution=bg_cfg.get("distribution", "normal"),
        prob=prob,

        # Spatial augmentation parameters
        rotation_range=bg_cfg.get("rotation_range", 10),
        scaling_range=bg_cfg.get("scaling_range", 0.1),
        shear_bounds=bg_cfg.get("shear_bounds", 0.005),
        translation_bounds=bg_cfg.get("translation_bounds", False),

        # Intensity augmentation parameters
        contrast_range=tuple(bg_cfg.get("contrast_range", [0.8, 1.2])),
        log_gamma_std=bg_cfg.get("log_gamma_std", 0.1),
        shift_offset=bg_cfg.get("shift_offset", 0.1),
        hist_control_points=bg_cfg.get("hist_control_points", 5),

        # Artefacts parameters
        noise_mean=bg_cfg.get("noise_mean", 0.02),
        noise_std=bg_cfg.get("noise_std", 0.015),
        rician_std=bg_cfg.get("rician_std", 0.01),
        gibbs_alpha=bg_cfg.get("gibbs_alpha", 0.4),
        blur_sigma=bg_cfg.get("blur_sigma", 0.25),
        bias_field_rng=tuple(bg_cfg.get("bias_field_rng", [0.0, 0.5])),
        
        # Motion artifacts
        motion_degrees=bg_cfg.get("motion_degrees", 3),
        motion_translation=bg_cfg.get("motion_translation", 5),
        motion_num_transforms=bg_cfg.get("motion_num_transforms", 4),
        ghost_num=tuple(bg_cfg.get("ghost_num", [1, 4])),
        ghost_intensity=tuple(bg_cfg.get("ghost_intensity", [0.1, 0.6])),
        torchio_noise_std=bg_cfg.get("torchio_noise_std", [0, 0.5]),

        # Resolution parameters
        min_res=bg_cfg.get("min_res", 0.8),
        max_res_iso=bg_cfg.get("max_res_iso", 2.0),
        max_res_aniso=bg_cfg.get("max_res_aniso", 2.0),
        atlas_res=bg_cfg.get("atlas_res", 1.0),
        thickness=bg_cfg.get("thickness", None),

        # Label config parameters
        generation_labels=GENERATION_LABELS,
        n_neutral_labels=N_NEUTRAL_LABELS,
        output_labels=None,

        # Toggle parameters
        use_sample=bg_cfg.get("use_sample", True),
        use_hemisphere_aware_flip=bg_cfg.get("use_hemisphere_aware_flip", True),
        use_dynamic_resolution=bg_cfg.get("use_dynamic_resolution", True),
        use_intensity_clip_normalize=bg_cfg.get("use_intensity_clip_normalize", True),
        n_channels=bg_cfg.get("n_channels", 1),
        use_specific_stats_for_channel=bg_cfg.get("use_specific_stats_for_channel", False),
        output_shape=tuple(bg_cfg.get("output_shape", [160, 192, 160])),
        use_random_cropping=bg_cfg.get("use_random_cropping", True),
        return_gradients=bg_cfg.get("return_gradients", False),
        return_segmentation=True,  # Return segmentation for visualization
        device=device,
    )

    # 5. ─── Read data paths from CSV files ───────────────────── #
    logger.info("Reading CSV files to get data paths...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv = Path(cfg.get("data.val_csv"))
    test_csv = Path(cfg.get("data.test_csv"))
    segmented_data_dir = Path(cfg.get("data.segmented_data_dir", ""))
    real_data_dir = Path(cfg.get("data.real_data_dir", ""))

    train_p, train_a, train_w, train_s, train_m = read_csv(
        train_csv,
        segmented_data_dir,
    )
    
    val_p, val_a, val_w, val_s, val_m = read_csv(
        val_csv,
        real_data_dir,
    )
    
    test_p, test_a, test_w, test_s, test_m = read_csv(
        test_csv,
        real_data_dir,
    )

    # 6. ─── Create datasets with appropriate transforms ──────── #
    logger.info("Creating datasets with transforms...")
    
    # Synthetic training dataset with brain generator
    train_synth_ds = BADataset(
        file_paths=train_p[:100],  # Limit to first 100 files
        age_labels=train_a[:100],
        sample_wts=train_w[:100] if train_w else None,
        sexes=train_s[:100] if train_s else None,
        modalities=train_m[:100] if train_m else None,
        transform=brain_generator,
        mode="train",
        cache_size=0,  # No cache for debugging
    )
    
    # Real validation dataset without augmentation
    val_ds = BADataset(
        file_paths=val_p[:50],  # Limit to first 50 files
        age_labels=val_a[:50],
        sexes=val_s[:50] if val_s else None,
        modalities=val_m[:50] if val_m else None,
        transform=None,  # No augmentation for validation
        mode="val",
        cache_size=0,
    )

    # 7. ─── Create dataloaders ─────────────────────────────── #
    logger.info("Setting up data loaders...")
    batch_size = cfg.get("training.batch_size", 4)
    
    dl_kwargs = dict(
        num_workers=2,  # Use fewer workers for debugging
        pin_memory=cfg.get("data.pin_memory", False),
        persistent_workers=False,  # Disable for debugging
    )
    
    # Training dataloader with synthetic data
    train_loader = DataLoader(
        train_synth_ds,
        batch_size=batch_size,
        shuffle=True,
        **dl_kwargs,
    )
    
    # Validation dataloader with real data
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )

    # 8. ─── Save data samples ───────────────────────────────── #
    logger.info("Sampling and saving data that would be fed to the model...")
    
    # Save config for reference
    cfg.save_config(output_dir / "config.yaml")
    
    # Determine save format
    save_format = cfg.get("spectate.save_format", "nifti")  # "nifti" or "numpy"
    
    # Save synthetic training batches
    train_metadata = save_dataset_batch(
        dataloader=train_loader,
        output_dir=output_dir / "synthetic_train",
        prefix="train",
        num_batches=3,
        save_format=save_format
    )
    
    # Save validation batches
    val_metadata = save_dataset_batch(
        dataloader=val_loader,
        output_dir=output_dir / "real_validation",
        prefix="val",
        num_batches=2,
        save_format=save_format
    )
    
    # 9. ─── Generate summary ────────────────────────────────── #
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config_file": cfg_file,
        "save_format": save_format,
        "brain_generator_config": bg_cfg,
        "output_directory": str(output_dir),
        "train_samples_saved": len(train_metadata),
        "val_samples_saved": len(val_metadata),
        "batch_size": batch_size,
    }
    
    summary_path = output_dir / "input_debug_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 10. ─── Print statistics about the data ──────────────────── #
    print("\n=== DATA STATISTICS ===")
    
    # Calculate statistics from the saved metadata
    if train_metadata:
        train_min = min(sample["image_min"] for sample in train_metadata)
        train_max = max(sample["image_max"] for sample in train_metadata)
        train_mean = np.mean([sample["image_mean"] for sample in train_metadata])
        train_std = np.mean([sample["image_std"] for sample in train_metadata])
        
        print(f"Synthetic Training Data:")
        print(f"  Min: {train_min:.4f}, Max: {train_max:.4f}")
        print(f"  Mean: {train_mean:.4f}, Std: {train_std:.4f}")
    
    if val_metadata:
        val_min = min(sample["image_min"] for sample in val_metadata)
        val_max = max(sample["image_max"] for sample in val_metadata)
        val_mean = np.mean([sample["image_mean"] for sample in val_metadata])
        val_std = np.mean([sample["image_std"] for sample in val_metadata])
        
        print(f"Real Validation Data:")
        print(f"  Min: {val_min:.4f}, Max: {val_max:.4f}")
        print(f"  Mean: {val_mean:.4f}, Std: {val_std:.4f}")
    
    logger.info("=== MODEL INPUT VISUALIZATION COMPLETE ===")
    logger.info(f"Images saved to: {output_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    
    print(f"\n🎉 Model input visualization complete!")
    print(f"📁 Check your images in: {output_dir}")
    print(f"📊 Summary: {summary_path}")
    print(f"💾 Format: {save_format}")


if __name__ == "__main__":
    sys.exit(main())