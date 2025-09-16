#!/usr/bin/env python
"""
Segmentation Input Spectating Script - Modified from train_seg_map.py
Saves segmentation map inputs (before one-hot encoding) from the dataset so you can inspect 
how spatial augmentation works on the raw segmentation labels.
Resource-optimized version with minimal memory footprint.
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
    Create a transform pipeline that applies spatial augmentations and label conversion
    but keeps the mapped labels as integers (no one-hot encoding).
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
        
        # Convert labels from original FreeSurfer labels to generation classes
        ConvertLabelsD(
            keys=["image"],
            generation_labels=GENERATION_LABELS,
            output_labels=GENERATION_CLASSES
        ),
        CastToTyped(keys=["image"], dtype=np.float32),
        
        # NO AsDiscreted - keep mapped labels as integers for visualization
    ]
    
    return Compose(spatial_transforms)


def create_original_transform():
    """Create transform that only does label conversion without spatial augmentation."""
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
        ConvertLabelsD(
            keys=["image"],
            generation_labels=GENERATION_LABELS,
            output_labels=GENERATION_CLASSES
        ),
        CastToTyped(keys=["image"], dtype=np.float32),
    ])


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
    """Analyze the label distribution in a segmentation tensor."""
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


def save_dataset_segmentations_efficient(dataset, output_dir: Path, prefix: str, num_images: int = 10, save_format: str = "nifti"):
    """
    Resource-efficient version: Save segmentation maps one by one to minimize memory usage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Saving {num_images} {prefix} segmentation maps to {output_dir} (format: {save_format}) ===")
    
    # Save metadata
    metadata = []
    
    for i in range(min(num_images, len(dataset))):
        try:
            # Load sample and immediately process to save memory
            sample = dataset[i]
            image = sample["image"]  # This should be the mapped segmentation labels
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
            
            # Clear variables to free memory
            del image, sample, seg_stats
            
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
        experiment_name = f'spectate_seg_inputs_{timestamp}'
    
    out_root = Path(cfg.get("output.output_dir", "output"))
    spectate_dir = out_root / "spectated_seg_inputs" / experiment_name
    log_dir = out_root / "logs" / experiment_name
    
    spectate_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("brain-age-spectate-seg", log_file=log_dir / "spectate_seg.log")
    
    logger.info("Initializing segmentation input spectating (resource-efficient mode)...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Experiment: {experiment_name}\nConfig: {cfg_file}")
    logger.info(f"Output directory: {spectate_dir}")

    # 3. ─── CSV → dataset (minimal data loading) ─────────── #
    logger.info("Reading CSV files...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv = Path(cfg.get("data.val_csv"))
    segmented_data_dir = Path(cfg.get("data.segmented_data_dir"))

    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w, train_s, train_m = read_csv(train_csv, segmented_data_dir)
    
    logger.info(f"Reading val CSV from {val_csv}")
    val_p, val_a, val_w, val_s, val_m = read_csv(val_csv, segmented_data_dir)
    
    # Limit data for efficiency - only take first 50 samples for each dataset
    train_p = train_p[:50]
    train_a = train_a[:50]
    train_w = train_w[:50] if train_w else None
    train_s = train_s[:50] if train_s else None
    train_m = train_m[:50] if train_m else None
    
    val_p = val_p[:20]
    val_a = val_a[:20]
    val_s = val_s[:20] if val_s else None
    val_m = val_m[:20] if val_m else None
    
    logger.info(f"Limited to: Train={len(train_p)}  Val={len(val_p)} for efficiency")

    # Print age ranges for reference
    print("=== AGE RANGES (LIMITED DATASET) ===")
    print(f"Train ages: min={min(train_a):.2f}, max={max(train_a):.2f}, mean={np.mean(train_a):.2f}")
    print(f"Val ages:   min={min(val_a):.2f}, max={max(val_a):.2f}, mean={np.mean(val_a):.2f}")

    # 4. ─── spatial transforms ─────────────────────────────── #
    logger.info("Initializing transforms...")

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

    # Create transforms
    augmented_transform = create_spatial_augmentation_only_transform(aug_config)
    original_transform = create_original_transform()

    logger.info(f"Spatial augmentation config: {aug_config.__dict__}")
    logger.info("NOTE: Using mapped segmentation labels (generation classes) for visualization")

    # 5. ─── create datasets (minimal resources) ─────────── #
    logger.info("Creating lightweight datasets...")
    
    # Training dataset WITH spatial augmentation
    train_ds_aug = BADataset(
        file_paths=train_p,
        age_labels=train_a,
        sample_wts=train_w,
        sexes=train_s,
        modalities=train_m,
        transform=augmented_transform,
        mode="train",
        cache_size=0,  # No cache to save memory
        clamp=False,
        normalize=False,
    )
    
    # Training dataset WITHOUT augmentation for comparison
    train_ds_orig = BADataset(
        file_paths=train_p,
        age_labels=train_a,
        sample_wts=train_w,
        sexes=train_s,
        modalities=train_m,
        transform=original_transform,
        mode="train",
        cache_size=0,  # No cache to save memory
        clamp=False,
        normalize=False,
    )
    
    # Validation dataset (no augmentation)
    val_ds = BADataset(
        file_paths=val_p,
        age_labels=val_a,
        sexes=val_s,
        modalities=val_m,
        transform=original_transform,
        mode="val",
        cache_size=0,  # No cache to save memory
        clamp=False,
        normalize=False,
    )

    # 6. ─── save segmentation maps (exactly 10 each) ──── #
    logger.info("Starting segmentation input spectating (10 images each type)...")
    
    # Save config for reference
    cfg.save_config(spectate_dir / "config.yaml")
    
    # Determine save format
    save_format = cfg.get("spectate.save_format", "nifti")
    logger.info(f"Saving segmentation maps in {save_format} format")
    
    # Save exactly 10 images of each type
    NUM_IMAGES = 10
    
    # Save training segmentations WITH spatial augmentation
    train_aug_metadata = save_dataset_segmentations_efficient(
        dataset=train_ds_aug,
        output_dir=spectate_dir / "train_augmented",
        prefix="train_aug",
        num_images=NUM_IMAGES,
        save_format=save_format
    )
    
    # Clear dataset to free memory
    del train_ds_aug
    
    # Save training segmentations WITHOUT augmentation (for comparison)
    train_orig_metadata = save_dataset_segmentations_efficient(
        dataset=train_ds_orig,
        output_dir=spectate_dir / "train_original", 
        prefix="train_orig",
        num_images=NUM_IMAGES,
        save_format=save_format
    )
    
    # Clear dataset to free memory
    del train_ds_orig
    
    # Save validation segmentations (no augmentation)
    val_metadata = save_dataset_segmentations_efficient(
        dataset=val_ds,
        output_dir=spectate_dir / "val_original",
        prefix="val",
        num_images=NUM_IMAGES,
        save_format=save_format
    )
    
    # Clear dataset to free memory
    del val_ds

    # 7. ─── create minimal analysis summary ─────────────── #
    logger.info("Creating minimal analysis summary...")
    
    def get_basic_stats(metadata_list, dataset_name):
        if not metadata_list:
            return {"dataset_name": dataset_name, "samples": 0}
        
        # Just get basic info without heavy computation
        all_labels = set()
        for sample_meta in metadata_list:
            labels = sample_meta["segmentation_stats"]["unique_labels"]
            all_labels.update(labels)
        
        return {
            "dataset_name": dataset_name,
            "samples": len(metadata_list),
            "unique_labels_found": sorted(list(all_labels)),
            "age_range": {
                "min": min(m["age"] for m in metadata_list),
                "max": max(m["age"] for m in metadata_list),
                "mean": sum(m["age"] for m in metadata_list) / len(metadata_list)
            }
        }
    
    analysis_summary = {
        "train_augmented": get_basic_stats(train_aug_metadata, "train_augmented"),
        "train_original": get_basic_stats(train_orig_metadata, "train_original"), 
        "val_original": get_basic_stats(val_metadata, "val_original"),
    }

    # 8. ─── final summary ────────────────────────────────── #
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config_file": cfg_file,
        "save_format": save_format,
        "data_type": "mapped_segmentation_labels",
        "images_per_type": NUM_IMAGES,
        "generation_labels": GENERATION_LABELS.tolist(),
        "generation_classes": GENERATION_CLASSES.tolist(),
        "spatial_augmentation_enabled": True,
        "one_hot_encoding_applied": False,
        "resource_optimized": True,
        "augmentation_config": {
            "probs": aug_config.probs,
            "params": aug_config.params,
        },
        "saved_counts": {
            "train_augmented": len(train_aug_metadata),
            "train_original": len(train_orig_metadata),
            "val_original": len(val_metadata),
        },
        "output_directory": str(spectate_dir),
        "basic_analysis": analysis_summary,
    }
    
    summary_path = spectate_dir / "spectate_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=== SEGMENTATION INPUT SPECTATING COMPLETE ===")
    logger.info(f"Segmentation maps saved to: {spectate_dir}")
    logger.info(f"Save format: {save_format}")
    logger.info(f"Images saved per type: {NUM_IMAGES}")
    logger.info(f"Training (augmented): {len(train_aug_metadata)}")
    logger.info(f"Training (original): {len(train_orig_metadata)}")
    logger.info(f"Validation: {len(val_metadata)}")
    logger.info(f"Summary saved to: {summary_path}")
    
    print(f"\n🎉 Segmentation input spectating complete!")
    print(f"📁 Check your segmentation maps in: {spectate_dir}")
    print(f"📊 Summary: {summary_path}")
    print(f"💾 Format: {save_format}")
    print(f"🧠 Data: Mapped segmentation labels (generation classes 0-14)")
    print(f"🔢 Images per type: {NUM_IMAGES}")
    print(f"⚡ Resource optimized: Minimal memory usage")
    print(f"🔄 Augmentation: Spatial transforms + label mapping applied")


if __name__ == "__main__":
    sys.exit(main())
