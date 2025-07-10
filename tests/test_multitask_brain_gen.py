#!/usr/bin/env python
"""
Visual test to verify that the generated brain image and segmentation ground truth
are spatially aligned after structural transformations.
"""

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

from brain_age_pred.brain_gen.brain_generator import BABrainGenerator
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES, GENERATION_LABELS, N_NEUTRAL_LABELS


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient between two binary masks.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()


def test_visual_alignment():
    """
    Generate a synthetic brain image and its segmentation ground truth,
    then visualize them to verify spatial alignment.
    """
    # ------------------------------------------------------------------
    # 1. SETUP: Load segmentation data
    # ------------------------------------------------------------------
    seg_path = Path(__file__).parent / "segmentation.nii.gz"
    assert seg_path.exists(), f"Test segmentation not found at {seg_path}"
    
    seg_img = nib.load(seg_path)
    seg_data = torch.from_numpy(seg_img.get_fdata(dtype=np.float32))[None, ...]  # Add channel dim
    
    print(f"Input segmentation shape: {seg_data.shape}")
    print(f"Unique labels in input: {torch.unique(seg_data).numpy()}")

    # ------------------------------------------------------------------
    # 2. BRAIN GENERATOR SETUP
    # ------------------------------------------------------------------
    n_classes = GENERATION_CLASSES.max() + 1
    
    # Prior distribution parameters (matching SynthSeg defaults)
    mean_loc, mean_scale = 125.0, 125.0
    std_loc, std_scale = 17.5, 17.5
    
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
    
    # Augmentation probabilities - enable some transforms for realistic test
    prob = {
        "flip": 0.8,
        "affine": 0.9,
        "contrast": 0.5,
        "gamma": 0.4,
        "scale_int": 0.3,
        "shift_int": 0.3,
        "hist_shift": 0.2,
        "noise": 0.3,
        "rician": 0.2,
        "gibbs": 0.1,
        "blur": 0.2,
        "bias": 0.3,
        "resolution": 0.1,
    }

    brain_gen = BABrainGenerator(
        prior_means=prior_means,
        prior_stds=prior_stds,
        distribution="uniform",
        prob=prob,
        return_segmentation=True,  # CRITICAL: get the seg_gt
        
        # Spatial augmentation parameters
        rotation_range=45,
        scaling_range=0.5,
        shear_bounds=0.05,
        translation_bounds=False,
        
        # Intensity augmentation parameters
        contrast_range=(0.7, 1.3),
        log_gamma_std=0.2,
        shift_offset=0.1,
        hist_control_points=8,
        
        # Artifacts parameters
        noise_mean=0.0,
        noise_std=0.02,
        rician_std=0.02,
        gibbs_alpha=0.5,
        blur_sigma=0.5,
        bias_field_rng=(0.0, 0.6),
        
        # Resolution parameters
        min_res=0.8,
        max_res_iso=2.0,
        max_res_aniso=2.0,
        atlas_res=1.0,
        
        # Output shape
        output_shape=(160, 192, 160),
        use_random_cropping=True,
    )

    # ------------------------------------------------------------------
    # 3. GENERATE SYNTHETIC BRAIN AND SEGMENTATION GT
    # ------------------------------------------------------------------
    torch.manual_seed(42)  # For reproducible results
    np.random.seed(42)
    
    result = brain_gen({"image": seg_data.clone()})
    
    synthetic_brain = result["image"].squeeze().cpu().numpy()  # Remove channel dim
    seg_gt = result["seg_gt"].squeeze().cpu().numpy()  # Remove channel dim
    
    print(f"Generated brain shape: {synthetic_brain.shape}")
    print(f"Segmentation GT shape: {seg_gt.shape}")
    print(f"Brain intensity range: [{synthetic_brain.min():.3f}, {synthetic_brain.max():.3f}]")
    print(f"Unique labels in seg_gt: {np.unique(seg_gt)}")

    # ------------------------------------------------------------------
    # 4. VISUAL COMPARISON: Plot middle slices
    # ------------------------------------------------------------------
    # Choose middle slices for visualization
    mid_axial = synthetic_brain.shape[0] // 2
    mid_sagittal = synthetic_brain.shape[1] // 2  
    mid_coronal = synthetic_brain.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Generated Brain vs Segmentation Ground Truth", fontsize=16)
    
    # Top row: Generated brain
    axes[0, 0].imshow(synthetic_brain[mid_axial, :, :], cmap='gray')
    axes[0, 0].set_title(f'Brain - Axial (slice {mid_axial})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(synthetic_brain[:, mid_sagittal, :], cmap='gray')
    axes[0, 1].set_title(f'Brain - Sagittal (slice {mid_sagittal})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(synthetic_brain[:, :, mid_coronal], cmap='gray')
    axes[0, 2].set_title(f'Brain - Coronal (slice {mid_coronal})')
    axes[0, 2].axis('off')
    
    # Bottom row: Segmentation ground truth
    axes[1, 0].imshow(seg_gt[mid_axial, :, :], cmap='tab20')
    axes[1, 0].set_title(f'Seg GT - Axial (slice {mid_axial})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(seg_gt[:, mid_sagittal, :], cmap='tab20')
    axes[1, 1].set_title(f'Seg GT - Sagittal (slice {mid_sagittal})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(seg_gt[:, :, mid_coronal], cmap='tab20')
    axes[1, 2].set_title(f'Seg GT - Coronal (slice {mid_coronal})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "brain_vs_segmentation.png", dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_dir / 'brain_vs_segmentation.png'}")
    
    # ------------------------------------------------------------------
    # 5. OVERLAY VISUALIZATION (FIXED CASTING ISSUE)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Brain with Segmentation Overlay", fontsize=16)
    
    # Create overlay by showing brain in grayscale and segmentation boundaries
    seg_boundaries = np.zeros_like(seg_gt, dtype=np.float64)  # Fixed: explicit float64 dtype
    
    for label in np.unique(seg_gt)[1:]:  # Skip background
        mask = (seg_gt == label).astype(np.float64)  # Fixed: explicit float64 dtype
        # Simple edge detection
        edges_x = np.abs(np.diff(mask, axis=1, prepend=0))
        edges_y = np.abs(np.diff(mask, axis=2, prepend=0))
        edges_z = np.abs(np.diff(mask, axis=0, prepend=0))
        edges = edges_x + edges_y + edges_z
        seg_boundaries += edges
    
    seg_boundaries = (seg_boundaries > 0).astype(np.float64)  # Fixed: explicit float64 dtype
    
    # Axial overlay
    brain_slice = synthetic_brain[mid_axial, :, :]
    seg_slice = seg_boundaries[mid_axial, :, :]
    axes[0].imshow(brain_slice, cmap='gray', alpha=0.8)
    axes[0].imshow(np.ma.masked_where(seg_slice == 0, seg_slice), cmap='Reds', alpha=0.8)
    axes[0].set_title(f'Overlay - Axial (slice {mid_axial})')
    axes[0].axis('off')
    
    # Sagittal overlay
    brain_slice = synthetic_brain[:, mid_sagittal, :]
    seg_slice = seg_boundaries[:, mid_sagittal, :]
    axes[1].imshow(brain_slice, cmap='gray', alpha=0.8)
    axes[1].imshow(np.ma.masked_where(seg_slice == 0, seg_slice), cmap='Reds', alpha=0.8)
    axes[1].set_title(f'Overlay - Sagittal (slice {mid_sagittal})')
    axes[1].axis('off')
    
    # Coronal overlay
    brain_slice = synthetic_brain[:, :, mid_coronal]
    seg_slice = seg_boundaries[:, :, mid_coronal]
    axes[2].imshow(brain_slice, cmap='gray', alpha=0.8)
    axes[2].imshow(np.ma.masked_where(seg_slice == 0, seg_slice), cmap='Reds', alpha=0.8)
    axes[2].set_title(f'Overlay - Coronal (slice {mid_coronal})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "brain_segmentation_overlay.png", dpi=300, bbox_inches='tight')
    print(f"Overlay visualization saved to: {output_dir / 'brain_segmentation_overlay.png'}")
    
    # ------------------------------------------------------------------
    # 6. PROPER SPATIAL ALIGNMENT TEST
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SPATIAL ALIGNMENT VERIFICATION")
    print("="*60)
    
    # Test 1: Check if both have the same background pattern
    brain_mask = (synthetic_brain > 0.01).astype(float)
    seg_mask = (seg_gt > 0).astype(float)
    
    # Calculate overlap metrics
    intersection = np.sum(brain_mask * seg_mask)
    union = np.sum((brain_mask + seg_mask) > 0)
    brain_total = np.sum(brain_mask)
    seg_total = np.sum(seg_mask)
    
    iou = intersection / union if union > 0 else 0
    brain_coverage = intersection / brain_total if brain_total > 0 else 0
    seg_coverage = intersection / seg_total if seg_total > 0 else 0
    
    print(f"IoU (Intersection over Union): {iou:.4f}")
    print(f"Brain coverage by segmentation: {brain_coverage:.4f}")
    print(f"Segmentation coverage by brain: {seg_coverage:.4f}")
    
    # Test 2: Check shape consistency
    print(f"\nShape consistency:")
    print(f"  Brain non-zero voxels: {brain_total:,}")
    print(f"  Segmentation non-zero voxels: {seg_total:,}")
    print(f"  Ratio: {brain_total/seg_total:.4f}")
    
    # Test 3: Visual assessment of major structures
    print(f"\nSegmentation label distribution:")
    unique_labels, counts = np.unique(seg_gt, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = 100 * count / seg_gt.size
        print(f"  Label {label:2d}: {count:7,} voxels ({percentage:5.2f}%)")
    
    # Test 4: Center of mass comparison (should be similar for aligned structures)
    from scipy import ndimage
    brain_com = ndimage.center_of_mass(brain_mask)
    seg_com = ndimage.center_of_mass(seg_mask)
    com_distance = np.sqrt(np.sum((np.array(brain_com) - np.array(seg_com))**2))
    
    print(f"\nCenter of mass comparison:")
    print(f"  Brain CoM: {brain_com}")
    print(f"  Seg CoM:   {seg_com}")
    print(f"  Distance:  {com_distance:.2f} voxels")
    
    # ------------------------------------------------------------------
    # 7. SHOW PLOTS
    # ------------------------------------------------------------------
    plt.show()
    
    print(f"\n✅ Visual test completed!")
    print(f"   - Generated brain shape: {synthetic_brain.shape}")
    print(f"   - Segmentation GT shape: {seg_gt.shape}")
    print(f"   - Spatial IoU: {iou:.4f}")
    print(f"   - Center of mass distance: {com_distance:.2f} voxels")
    
    # Final assessment
    if iou > 0.8 and com_distance < 5.0:
        print("✅ PASS: Spatial alignment looks good!")
    elif iou > 0.6:
        print("⚠️  PARTIAL: Some spatial alignment, but could be better")
    else:
        print("❌ FAIL: Poor spatial alignment detected")
    
    return synthetic_brain, seg_gt, iou


if __name__ == "__main__":
    test_visual_alignment()
