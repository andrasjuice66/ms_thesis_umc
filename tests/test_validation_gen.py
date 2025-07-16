#!/usr/bin/env python
"""
Test script for ValidationGenerator to see what it outputs.
Loads a real image and its corresponding segmentation, processes them,
and saves the results.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from brain_age_pred.brain_gen.validation_generator import ValidationGenerator
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES, GENERATION_LABELS, N_NEUTRAL_LABELS
import torch
from monai.transforms import CenterSpatialCropd

# ------------------------------------------------------------------
# 1) INPUT IMAGE (real MRI scan)
# ------------------------------------------------------------------
# For testing, we'll use the segmentation as if it were a real image
IMAGE_PATH = "/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/brain_age_pred/tests/test_dir/segmentation.nii.gz"
SEGMENTED_DATA_DIR = "/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/brain_age_pred/tests/"  # Directory containing segmentations

# Load the "image" (using segmentation file as example)
img_nii = nib.load(IMAGE_PATH)
img_data = img_nii.get_fdata().astype(np.float32)

print(f"Original image shape: {img_data.shape}")
print(f"Original image type: {type(img_data)}")

# Ensure it's really 3D and add channel dimension
if img_data.ndim == 3:
    img_data = img_data[None, ...]  # Add channel dimension: (1, D, H, W)
    print(f"After adding channel dim: {img_data.shape}")

# Convert to tensor
img_tensor = torch.from_numpy(img_data).float()

# ------------------------------------------------------------------
# 2) BUILD VALIDATION GENERATOR
# ------------------------------------------------------------------
validation_gen = ValidationGenerator(
    segmented_data_dir=SEGMENTED_DATA_DIR,
    generation_labels=GENERATION_LABELS,      # Original FreeSurfer labels
    output_labels=GENERATION_LABELS,          # Keep original labels (no conversion)
    use_intensity_clip_normalize=True,        # Apply intensity normalization
    return_segmentation=True,                 # Return segmentation
)

center_crop = CenterSpatialCropd(keys=["image", "seg_gt"], roi_size=(160, 192, 160), allow_missing_keys=True)


print(f"ValidationGenerator configured:")
print(f"  - Segmented data dir: {SEGMENTED_DATA_DIR}")
print(f"  - Generation labels: {GENERATION_LABELS[:10]}...")  # Show first 10
print(f"  - Output labels: {validation_gen.output_labels[:10]}...")
print(f"  - Return segmentation: {validation_gen.return_segmentation}")

# ------------------------------------------------------------------
# 3) OUTPUT FOLDER
# ------------------------------------------------------------------
out_dir = Path("/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/brain_age_pred/tests/validation_gen_images")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 4) PROCESS & SAVE SAMPLES
# ------------------------------------------------------------------
for k in range(5):  # Generate fewer since it's just processing, not generating
    # Create sample with image and path metadata
    sample = {
        "image": img_tensor.clone(),  # Clone to avoid modifying original
        "__image_path__": IMAGE_PATH  # Path info for finding corresponding segmentation
    }
    
    try:
        # Process through validation generator
        out_dict = validation_gen(sample)
        out_dict = center_crop(out_dict)
        


        
        # Extract outputs
        processed_img = out_dict["image"].squeeze(0).cpu().numpy()
        processed_seg = out_dict["seg_gt"].squeeze(0).cpu().numpy()
        
        print(f"\nSample {k}:")
        print(f"  Processed image shape: {processed_img.shape}")
        print(f"  Processed image range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")
        print(f"  Processed seg shape: {processed_seg.shape}")
        # print(f"  Unique seg labels: {np.unique(processed_seg)}")
        
        # Save processed image
        img_nifti = nib.Nifti1Image(processed_img, affine=img_nii.affine, header=img_nii.header)
        img_fname = out_dir / f"processed_image_{k:02d}.nii.gz"
        nib.save(img_nifti, img_fname)
        print(f"  Saved processed image → {img_fname}")
        
        # Save processed segmentation
        seg_nifti = nib.Nifti1Image(processed_seg, affine=img_nii.affine, header=img_nii.header)
        seg_fname = out_dir / f"processed_seg_{k:02d}.nii.gz"
        nib.save(seg_nifti, seg_fname)
        print(f"  Saved processed seg → {seg_fname}")
        
    except Exception as e:
        print(f"Error processing sample {k}: {e}")
        continue

print(f"\nDone. Processed files are in: {out_dir.resolve()}")

# ------------------------------------------------------------------
# 5) COMPARE INPUT VS OUTPUT
# ------------------------------------------------------------------
print("\n" + "="*60)
print("COMPARISON: INPUT vs OUTPUT")
print("="*60)

# Load one of the outputs for comparison
if (out_dir / "processed_image_00.nii.gz").exists():
    processed_img = nib.load(out_dir / "processed_image_00.nii.gz").get_fdata()
    processed_seg = nib.load(out_dir / "processed_seg_00.nii.gz").get_fdata()
    
    print(f"Original image range: [{img_data.squeeze().min():.3f}, {img_data.squeeze().max():.3f}]")
    print(f"Processed image range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")
    print(f"Original unique labels: {np.unique(img_data.squeeze())}")
    print(f"Processed unique labels: {np.unique(processed_seg)}")
    
    # Check if labels were converted
    original_labels = set(np.unique(img_data.squeeze()))
    processed_labels = set(np.unique(processed_seg))
    
    if original_labels == processed_labels:
        print("✓ Labels preserved (no conversion)")
    else:
        print("⚠ Labels were converted:")
        print(f"  Original had: {len(original_labels)} unique labels")
        print(f"  Processed has: {len(processed_labels)} unique labels")
        print(f"  Added labels: {processed_labels - original_labels}")
        print(f"  Removed labels: {original_labels - processed_labels}")