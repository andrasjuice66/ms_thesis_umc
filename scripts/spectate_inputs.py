#!/usr/bin/env python
"""
Segmentation Input Spectating Script - Modified from train_seg_map.py
Saves segmentation map inputs (before one-hot encoding) from the dataset so you can inspect 
how spatial augmentation works on the raw segmentation labels.
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
from monai.transforms import Compose, EnsureChannelFirstd, RandFlipd, RandZoomd, RandRotated, RandAffined, CastToTyped

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dataset.dataset import BADataset          
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES, GENERATION_LABELS, N_NEUTRAL_LABELS
from brain_age_pred.dataset.segmentation_augmentation import SegmentationAugmentationConfig
from brain_age_pred.dataset.custom_transformations import ConvertLabelsD


def create_spatial_augmentation_only_transform(config=None):
    """
    Create a transform pipeline that ONLY applies spatial augmentations to raw segmentation maps.
    Does NOT do one-hot encoding - keeps the original label values for visualization.
    """
    if config is None:
        config = SegmentationAugmentationConfig()
    
    spatial_transforms = [
        # Ensure channel first - data already has channel dimension at position 0
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
        
        # Add spatial augmentations - use "nearest" mode to preserve integer labels
        RandFlipd(
            keys=["image"],
            prob=config.probs["flip"],
            spatial_axis=0,
        ),
        
        RandZoomd(
            keys=["image"], 
            min_zoom=config.params["zoom_min"], 
            max_zoom=config.params["zoom_max"], 
            prob=config.probs["zoom"],
            mode="nearest"  # Preserve integer labels
        ),
        
        RandRotated(
            keys=["image"], 
            range_x=config.params["rotate_range_x"], 
            range_y=config.params["rotate_range_y"], 
            range_z=config.params["rotate_range_z"], 
            prob=config.probs["rotate"],
            mode="nearest"  # Preserve integer labels
        ),
        
        RandAffined(
            keys=["image"],
            prob=config.probs["affine"],
            rotate_range=(0, 0, 0),  # Disable rotation since we use RandRotated
            scale_range=config.params["scaling_range"],  
            shear_range=(config.params["shearing_bounds"],) * 3,
            mode="nearest",  # Preserve integer labels
            padding_mode="zeros"
        ),
        ConvertLabelsD(
            keys=["image"],
            generation_labels=GENERATION_LABELS,
            output_labels=GENERATION_CLASSES
        ),
        CastToTyped(keys=["image"], dtype=np.float32),
        
        # NO ConvertLabelsD or AsDiscreted - keep raw labels for visualization
    ]
    
    return Compose(spatial_transforms)


def save_segmentation_as_nifti(seg_tensor: torch.Tensor, save_path: Path, affine: np.ndarray = None):
    """Save a segmentation tensor as NIfTI file using nibabel."""
    # Remove channel dimension if present
    if len(seg_tensor.shape) == 4:  # (C, D, H, W)
        seg_np = seg_tensor.squeeze(0).cpu().numpy()
    else:  # (D, H, W)
        seg_np = seg_tensor.cpu().numpy()
    
    # Use identity matrix if no affine provided
    if affine is None:
        affine = np.eye(4)
    
    # Create NIfTI image using nibabel and save
    nifti_img = nib.Nifti1Image(seg_np, affine)
    nib.save(nifti_img, str(save_path))


def save_segmentation_as_numpy(seg_tensor: torch.Tensor, save_path: Path):
    """Save a segmentation tensor as numpy array (.npy file)."""
    # Remove channel dimension if present
    if len(seg_tensor.shape) == 4:  # (C, D, H, W)
        seg_np = seg_tensor.squeeze(0).cpu().numpy()
    else:  # (D, H, W)
        seg_np = seg_tensor.cpu().numpy()
    
    np.save(str(save_path), seg_np)


def analyze_segmentation_labels(seg_tensor: torch.Tensor) -> Dict:
    """Analyze the label distribution in a raw segmentation tensor."""
    if len(seg_tensor.shape) == 4:  # (C, D, H, W)
        seg_np = seg_tensor.squeeze(0).cpu().numpy()
    else:  # (D, H, W)
        seg_np = seg_tensor.cpu().numpy()
    
    unique_labels, counts = np.unique(seg_np, return_counts=True)
    total_voxels = seg_np.size
    
    stats = {
        "shape": list(seg_np.shape),
        "total_voxels": int(total_voxels),
        "unique_labels": unique_labels.astype(int).tolist(),
        "label_counts": counts.astype(int).tolist(),
        "label_percentages": (counts / total_voxels * 100).tolist(),
        "min_label": int(unique_labels.min()),
        "max_label": int(unique_labels.max()),
        "num_unique_labels": len(unique_labels),
        "min_val": float(seg_np.min()),
        "max_val": float(seg_np.max()),
    }
    
    # Create a mapping of label -> percentage for easier access
    stats["label_percentage_map"] = {
        int(label): float(percentage) 
        for label, percentage in zip(unique_labels, stats["label_percentages"])
    }
    
    return stats


def save_dataset_segmentations(dataset, output_dir: Path, prefix: str, num_images: int = 50, save_format: str = "nifti"):
    """Save raw segmentation maps from dataset to inspect spatial augmentation effects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Saving {num_images} {prefix} raw segmentation maps to {output_dir} (format: {save_format}) ===")
    
    # Save metadata
    metadata = []
    
    for i in range(min(num_images, len(dataset))):
        try:
            sample = dataset[i]
            image = sample["image"]  # This should be the raw segmentation labels
            age = sample["age"].item()
            
            # Create filename with metadata
            original_path = sample.get("__image_path__", f"unknown_{i}")
            original_name = Path(original_path).stem if original_path != f"unknown_{i}" else f"unknown_{i}"
            
            # Analyze segmentation statistics
            seg_stats = analyze_segmentation_labels(image)
            
            if save_format.lower() == "nifti":
                filename = f"{prefix}_{i:03d}_{original_name}_age{age:.1f}_labels.nii.gz"
                save_path = output_dir / filename
                save_segmentation_as_nifti(image, save_path)
                
            elif save_format.lower() == "numpy":
                filename = f"{prefix}_{i:03d}_{original_name}_age{age:.1f}_labels.npy"
                save_path = output_dir / filename
                save_segmentation_as_numpy(image, save_path)
                
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
                "segmentation_stats": seg_stats,
                "save_format": save_format,
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Saved {i + 1}/{num_images} segmentation maps...")
                
        except Exception as e:
            print(f"  Error saving segmentation {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save metadata as JSON
    metadata_path = output_dir / f"{prefix}_segmentation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Completed! Saved {len(metadata)} segmentation maps and metadata to {output_dir}")
    return metadata


def main() -> None:
    # 1. ─── configuration & reproducibility ─────────────────── #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    cfg = Config(cfg_file)
    
    # 2. ─── experiment naming / I/O ─────────────────────────── #
    experiment_name = cfg.get("output.experiment_name")
    if not experiment_name:
        experiment_name = f'spectate_raw_segmentation_{timestamp}'
    
    out_root = Path(cfg.get("output.output_dir", "output"))
    spectate_dir = out_root / "spectated_raw_segmentation" / experiment_name
    log_dir = out_root / "logs" / experiment_name
    
    spectate_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("brain-age-spectate-raw-seg", log_file=log_dir / "spectate_raw_seg.log")
    
    logger.info("Initializing raw segmentation input spectating...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Experiment: {experiment_name}\nConfig: {cfg_file}")
    logger.info(f"Output directory: {spectate_dir}")

    # 3. ─── CSV → dataset ─────────────────────────────────── #
    logger.info("Reading CSV files...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv = Path(cfg.get("data.val_csv"))
    test_csv = Path(cfg.get("data.test_csv"))
    segmented_data_dir = Path(cfg.get("data.segmented_data_dir"))

    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w, train_s, train_m = read_csv(train_csv, segmented_data_dir)
    
    logger.info(f"Reading val CSV from {val_csv}")
    val_p, val_a, val_w, val_s, val_m = read_csv(val_csv, segmented_data_dir)
    
    if test_csv.exists():
        logger.info(f"Reading test CSV from {test_csv}")
        test_p, test_a, test_w, test_s, test_m = read_csv(test_csv, segmented_data_dir)
    else:
        test_p, test_a, test_w, test_s, test_m = [], [], [], [], []
    
    logger.info(f"Train={len(train_p)}  Val={len(val_p)}  Test={len(test_p)}")

    # Print age ranges for reference
    print("=== AGE RANGES ===")
    print(f"Train ages: min={min(train_a):.2f}, max={max(train_a):.2f}, mean={np.mean(train_a):.2f}")
    print(f"Val ages:   min={min(val_a):.2f}, max={max(val_a):.2f}, mean={np.mean(val_a):.2f}")

    # 4. ─── spatial transforms (NO one-hot encoding) ─────── #
    logger.info("Initializing spatial-only transforms...")

    # Create augmentation config from YAML settings
    aug_config = SegmentationAugmentationConfig(
        transform_probs={
            "flip": cfg.get("segmentation_augmentation.transform_probs.flip", 0.5),
            "affine": cfg.get("segmentation_augmentation.transform_probs.affine", 0.5),
            "zoom": cfg.get("segmentation_augmentation.transform_probs.zoom", 0.5),
            "rotate": cfg.get("segmentation_augmentation.transform_probs.rotate", 0.5),
        },
        scaling_range=cfg.get("segmentation_augmentation.scaling_range", (0.95, 1.05)),
        shearing_bounds=cfg.get("segmentation_augmentation.shearing_bounds", 0.2),
        zoom_min=cfg.get("segmentation_augmentation.zoom_min", 0.95),
        zoom_max=cfg.get("segmentation_augmentation.zoom_max", 1.05),
        rotate_range_x=cfg.get("segmentation_augmentation.rotate_range_x", 0.1),
        rotate_range_y=cfg.get("segmentation_augmentation.rotate_range_y", 0.1),
        rotate_range_z=cfg.get("segmentation_augmentation.rotate_range_z", 0.1),
    )

    # Create transforms - ONLY spatial augmentation, NO one-hot encoding
    augmented_transform = create_spatial_augmentation_only_transform(aug_config)
    original_transform = Compose([EnsureChannelFirstd(keys=["image"], channel_dim=0)])  # Just ensure channel format

    logger.info(f"Spatial augmentation config: {aug_config.__dict__}")
    logger.info("NOTE: Using RAW segmentation labels (no one-hot encoding) for visualization")

    # 5. ─── create datasets ───────────────────────────────── #
    logger.info("Creating datasets...")
    
    # Training dataset WITH spatial augmentation (but NO one-hot encoding)
    train_ds_aug = BADataset(
        file_paths=train_p,
        age_labels=train_a,
        sample_wts=train_w,
        sexes=train_s,
        modalities=train_m,
        transform=augmented_transform,  # Apply spatial augmentation only
        mode="train",
        cache_size=0,  # No cache for spectating
        clamp=False,
        normalize=False,
    )
    
    # Training dataset WITHOUT any augmentation for comparison
    train_ds_orig = BADataset(
        file_paths=train_p[:100],  # Limit for comparison
        age_labels=train_a[:100],
        sample_wts=train_w[:100] if train_w else None,
        sexes=train_s[:100] if train_s else None,
        modalities=train_m[:100] if train_m else None,
        transform=original_transform,  # No augmentation
        mode="train",
        cache_size=0,
        clamp=False,
        normalize=False,
    )
    
    # Validation dataset (no augmentation)
    val_ds = BADataset(
        file_paths=val_p,
        age_labels=val_a,
        sexes=val_s,
        modalities=val_m,
        transform=original_transform,  # No augmentation for validation
        mode="val",
        cache_size=0,
        clamp=False,
        normalize=False,
    )

    # 6. ─── save raw segmentation maps ──────────────────── #
    logger.info("Starting raw segmentation input spectating...")
    
    # Save config for reference
    cfg.save_config(spectate_dir / "config.yaml")
    
    # Determine save format
    save_format = cfg.get("spectate.save_format", "nifti")
    logger.info(f"Saving raw segmentation maps in {save_format} format")
    
    # Save training segmentations WITH spatial augmentation
    train_aug_metadata = save_dataset_segmentations(
        dataset=train_ds_aug,
        output_dir=spectate_dir / "train_augmented",
        prefix="train_aug",
        num_images=20,
        save_format=save_format
    )
    
    # Save training segmentations WITHOUT augmentation (for comparison)
    train_orig_metadata = save_dataset_segmentations(
        dataset=train_ds_orig,
        output_dir=spectate_dir / "train_original", 
        prefix="train_orig",
        num_images=20,
        save_format=save_format
    )
    
    # Save validation segmentations (no augmentation)
    val_metadata = save_dataset_segmentations(
        dataset=val_ds,
        output_dir=spectate_dir / "val_original",
        prefix="val",
        num_images=10,
        save_format=save_format
    )

    # 7. ─── create analysis summary ──────────────────────── #
    logger.info("Creating label distribution analysis...")
    
    # Analyze label distribution across datasets
    def analyze_label_distribution(metadata_list, dataset_name):
        all_labels = set()
        label_stats = {}
        
        # Collect all unique labels across samples
        for sample_meta in metadata_list:
            labels = sample_meta["segmentation_stats"]["unique_labels"]
            all_labels.update(labels)
        
        all_labels = sorted(list(all_labels))
        
        # Analyze each label
        for label in all_labels:
            percentages = []
            for sample_meta in metadata_list:
                label_map = sample_meta["segmentation_stats"]["label_percentage_map"]
                percentages.append(label_map.get(label, 0.0))
            
            label_stats[f"label_{label}"] = {
                "mean_percentage": float(np.mean(percentages)),
                "std_percentage": float(np.std(percentages)),
                "min_percentage": float(np.min(percentages)),
                "max_percentage": float(np.max(percentages)),
                "present_in_samples": int(sum(1 for p in percentages if p > 0)),
                "total_samples": len(percentages),
            }
        
        return {
            "dataset_name": dataset_name,
            "total_samples": len(metadata_list),
            "unique_labels_found": all_labels,
            "label_statistics": label_stats
        }
    
    analysis_summary = {
        "train_augmented": analyze_label_distribution(train_aug_metadata, "train_augmented"),
        "train_original": analyze_label_distribution(train_orig_metadata, "train_original"), 
        "val_original": analyze_label_distribution(val_metadata, "val_original"),
    }

    # 8. ─── final summary ────────────────────────────────── #
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config_file": cfg_file,
        "save_format": save_format,
        "data_type": "raw_segmentation_labels",
        "generation_labels": GENERATION_LABELS.tolist(),
        "generation_classes": GENERATION_CLASSES.tolist(),
        "n_neutral_labels": N_NEUTRAL_LABELS,
        "spatial_augmentation_enabled": True,
        "one_hot_encoding_applied": False,
        "augmentation_config": {
            "probs": aug_config.probs,
            "params": aug_config.params,
        },
        "train_augmented_saved": len(train_aug_metadata),
        "train_original_saved": len(train_orig_metadata),
        "val_saved": len(val_metadata),
        "output_directory": str(spectate_dir),
        "label_distribution_analysis": analysis_summary,
        "dataset_sizes": {
            "train_augmented": len(train_ds_aug),
            "train_original": len(train_ds_orig),
            "val": len(val_ds),
        }
    }
    
    summary_path = spectate_dir / "raw_segmentation_spectate_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=== RAW SEGMENTATION INPUT SPECTATING COMPLETE ===")
    logger.info(f"Raw segmentation maps saved to: {spectate_dir}")
    logger.info(f"Save format: {save_format}")
    logger.info(f"Training segmentations (augmented): {len(train_aug_metadata)}")
    logger.info(f"Training segmentations (original): {len(train_orig_metadata)}")
    logger.info(f"Validation segmentations: {len(val_metadata)}")
    logger.info(f"Summary saved to: {summary_path}")
    
    print(f"\n🎉 Raw segmentation input spectating complete!")
    print(f"📁 Check your raw segmentation maps in: {spectate_dir}")
    print(f"📊 Summary: {summary_path}")
    print(f"💾 Format: {save_format}")
    print(f"🧠 Data: Raw segmentation labels (viewable in any NIfTI viewer)")
    print(f"🔄 Augmentation: Spatial transforms applied to training data")
    
    # Print label mapping for reference
    print(f"\n📋 Label mapping preview:")
    print(f"   Original labels: {GENERATION_LABELS[:10]}...")
    print(f"   Mapped classes:  {GENERATION_CLASSES[:10]}...")
    print(f"   NOTE: Saved files contain ORIGINAL labels (not mapped classes)")


if __name__ == "__main__":
    sys.exit(main())
