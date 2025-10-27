#!/usr/bin/env python
"""
Image Spectating Script - Modified from training script
Saves augmented images from the dataset so you can inspect how augmentation works.
"""
import os, sys, time, json, random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import multiprocessing as mp

import pandas as pd
import numpy as np
import torch
import nibabel as nib

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dataset.dataset import BADataset          
from brain_age_pred.dataset.augmentation import AugmentationPipeline
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv
from brain_age_pred.dataset.brainagenext_transforms import get_train_transforms, get_val_transforms


def save_image_as_nifti(image_tensor: torch.Tensor, save_path: Path, affine: np.ndarray = None):
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


def save_image_as_numpy(image_tensor: torch.Tensor, save_path: Path):
    """Save a torch tensor as numpy array (.npy file)."""
    # Convert tensor to numpy and remove channel dimension if present
    if len(image_tensor.shape) == 4:  # (C, D, H, W)
        image_np = image_tensor.squeeze(0).cpu().numpy()
    else:  # (D, H, W)
        image_np = image_tensor.cpu().numpy()
    
    # Save as numpy array
    np.save(str(save_path), image_np)


def save_dataset_images(dataset, output_dir: Path, prefix: str, num_images: int = 100, save_format: str = "nifti"):
    """Save images from dataset to inspect augmentation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Saving {num_images} {prefix} images to {output_dir} (format: {save_format}) ===")
    
    # Save metadata
    metadata = []
    
    for i in range(min(num_images, len(dataset))):
        try:
            sample = dataset[i]
            image = sample["image"]
            age = sample["age"].item()
            
            # Create filename with metadata
            original_path = sample.get("__image_path__", f"unknown_{i}")
            original_name = Path(original_path).stem if original_path != f"unknown_{i}" else f"unknown_{i}"
            
            if save_format.lower() == "nifti":
                filename = f"{prefix}_{i:03d}_{original_name}_age{age:.1f}.nii.gz"
                save_path = output_dir / filename
                save_image_as_nifti(image, save_path)
            elif save_format.lower() == "numpy":
                filename = f"{prefix}_{i:03d}_{original_name}_age{age:.1f}.npy"
                save_path = output_dir / filename
                save_image_as_numpy(image, save_path)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
            
            # Store metadata
            metadata.append({
                "index": i,
                "filename": filename,
                "age": age,
                "original_path": original_path,
                "sex": sample.get("sex", "unknown"),
                "modality": sample.get("modality", "unknown"),
                "image_shape": list(image.shape),
                "image_min": float(image.min()),
                "image_max": float(image.max()),
                "image_mean": float(image.mean()),
                "image_std": float(image.std()),
                "save_format": save_format,
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Saved {i + 1}/{num_images} images...")
                
        except Exception as e:
            print(f"  Error saving image {i}: {e}")
            continue
    
    # Save metadata as JSON
    metadata_path = output_dir / f"{prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Completed! Saved {len(metadata)} images and metadata to {output_dir}")
    return metadata


def main() -> None:
    # 1. ─── configuration & reproducibility ─────────────────── #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    cfg = Config(cfg_file)
    
    # 2. ─── experiment naming / I/O ─────────────────────────── #
    experiment_name = cfg.get("output.experiment_name")
    if not experiment_name:
        experiment_name = f'spectate_images_{timestamp}'
    
    out_root = Path(cfg.get("output.output_dir", "output"))
    spectate_dir = out_root / "spectated_images" / experiment_name
    log_dir = out_root / "logs" / experiment_name
    
    spectate_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("brain-age-spectate", log_file=log_dir / "spectate.log")
    
    logger.info("Initializing image spectating...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Experiment: {experiment_name}\nConfig: {cfg_file}")
    logger.info(f"Output directory: {spectate_dir}")

    # 3. ─── transforms (same as training) ─────────────────── #
    logger.info("Initializing augmentation transforms...")
    aug_cfg = cfg.get("augmentation", {})
    if aug_cfg.get("use_augmentation", False):
        transform = AugmentationPipeline(**aug_cfg)
        logger.info(f"Augmentation enabled with config: {aug_cfg}")
    else:
        transform = None
        logger.info("Augmentation disabled")

    # 4. ─── CSV → dataset ─────────────────────────────────── #
    logger.info("Reading CSV files...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv = Path(cfg.get("data.val_csv"))
    test_csv = Path(cfg.get("data.test_csv"))
    real_data_dir = Path(cfg.get("data.real_data_dir"))

    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w, train_s, train_m = read_csv(train_csv, real_data_dir)
    
    logger.info(f"Reading val CSV from {val_csv}")
    val_p, val_a, val_w, val_s, val_m = read_csv(val_csv, real_data_dir)
    
    logger.info(f"Train={len(train_p)}  Val={len(val_p)}")

    # Print age ranges for reference
    print("=== AGE RANGES ===")
    print(f"Train ages: min={min(train_a):.2f}, max={max(train_a):.2f}, mean={np.mean(train_a):.2f}")
    print(f"Val ages:   min={min(val_a):.2f}, max={max(val_a):.2f}, mean={np.mean(val_a):.2f}")

    # 5. ─── create datasets ───────────────────────────────── #
    logger.info("Creating datasets...")

    train_transform = get_train_transforms(
    image_key="image",                 # MUST match BADataset's key
    pixdim=(1, 1, 1),
    spatial_pad=(160, 192, 160),       # optional
    center_crop=(160, 192, 160),       # optional
    crop_foreground=True,)

    val_transform = get_val_transforms(
    image_key="image",
    pixdim=(1, 1, 1),
    spatial_pad=(160, 192, 160),
    center_crop=(160, 192, 160),
    crop_foreground=True,)
    
    # Training dataset WITH augmentation
    train_ds = BADataset(
        file_paths=train_p,
        age_labels=train_a,
        sample_wts=train_w,
        sexes=train_s,
        modalities=train_m,
        transform=train_transform,  # Apply augmentation
        mode="train",
        cache_size=0,  # No cache for spectating
    )
    
    # Validation dataset WITHOUT augmentation  
    val_ds = BADataset(
        file_paths=val_p,
        age_labels=val_a,
        sexes=val_s,
        modalities=val_m,
        transform=val_transform,  # No augmentation for validation
        mode="val",
        cache_size=0,
    )

    # 6. ─── save images ───────────────────────────────────── #
    logger.info("Starting image spectating...")
    
    # Save config for reference
    cfg.save_config(spectate_dir / "config.yaml")
    
    # Determine save format (can be configured or default to nifti)
    save_format = cfg.get("spectate.save_format", "nifti")  # "nifti" or "numpy"
    logger.info(f"Saving images in {save_format} format")
    
    # Save training images (with augmentation)
    train_metadata = save_dataset_images(
        dataset=train_ds,
        output_dir=spectate_dir / "train_augmented",
        prefix="train",
        num_images=10,
        save_format=save_format
    )
    
    # Save validation images (without augmentation) 
    val_metadata = save_dataset_images(
        dataset=val_ds,
        output_dir=spectate_dir / "val_original", 
        prefix="val",
        num_images=10,  # Fewer validation images
        save_format=save_format
    )
    
    # If augmentation is enabled, also save some training images WITHOUT augmentation for comparison
    if transform is not None:
        logger.info("Creating comparison dataset without augmentation...")
        train_ds_no_aug = BADataset(
            file_paths=train_p[:50],  # Just first 50 for comparison
            age_labels=train_a[:50],
            sample_wts=train_w[:50] if train_w else None,
            sexes=train_s[:50] if train_s else None,
            modalities=train_m[:50] if train_m else None,
            transform=None,  # No augmentation
            mode="train",
            cache_size=0,
        )
        
        comparison_metadata = save_dataset_images(
            dataset=train_ds_no_aug,
            output_dir=spectate_dir / "train_original",
            prefix="train_orig",
            num_images=10,
            save_format=save_format
        )

    # 7. ─── summary ───────────────────────────────────────── #
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config_file": cfg_file,
        "save_format": save_format,
        "augmentation_enabled": aug_cfg.get("use_augmentation", False),
        "augmentation_config": aug_cfg,
        "train_images_saved": len(train_metadata),
        "val_images_saved": len(val_metadata),
        "output_directory": str(spectate_dir),
        "train_dataset_size": len(train_ds),
        "val_dataset_size": len(val_ds),
    }
    
    summary_path = spectate_dir / "spectate_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=== SPECTATING COMPLETE ===")
    logger.info(f"Images saved to: {spectate_dir}")
    logger.info(f"Save format: {save_format}")
    logger.info(f"Training images (augmented): {len(train_metadata)}")
    logger.info(f"Validation images (original): {len(val_metadata)}")
    if transform is not None:
        logger.info("Comparison images (original): 50")
    logger.info(f"Summary saved to: {summary_path}")
    
    print(f"\n🎉 Image spectating complete!")
    print(f"📁 Check your images in: {spectate_dir}")
    print(f"📊 Summary: {summary_path}")
    print(f"💾 Format: {save_format}")


if __name__ == "__main__":
    sys.exit(main())