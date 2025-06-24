#!/usr/bin/env python
"""
Generate 5 synthetic MR volumes from one SynthSeg segmentation,
using a **uniform** hyper-prior (same as the original SynthSeg paper),
and save them as NIfTI files in  ./brain_gen_images/.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from brain_age_pred.brain_gen.brain_generator import BABrainGenerator
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES, GENERATION_LABELS, N_NEUTRAL_LABELS
import torch

# ------------------------------------------------------------------
# 1) INPUT SEGMENTATION (integer labels in SynthSeg space)
# ------------------------------------------------------------------
SEG_PATH = "C:/Projects/thesis_project/brain_age_pred/tests/segmentation.nii.gz"              # <-- change to an existing file
seg_img  = nib.load(SEG_PATH)
seg_data = seg_img.get_fdata().astype(np.int16)          # numpy array [D,H,W]

# Add debug prints
print(f"Original seg_data shape: {seg_data.shape}")
print(f"Original seg_data type: {type(seg_data)}")

# Ensure it's really 3D and add channel dimension
if seg_data.ndim == 3:
    seg_data = seg_data[None, ...]  # Add channel dimension: (1, D, H, W)
    print(f"After adding channel dim: {seg_data.shape}")

# Add device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert to tensor and move to device
seg_data = torch.from_numpy(seg_data).float().to(device)

# ------------------------------------------------------------------
# 2) BUILD A GENERATOR  –  UNIFORM distributions
#    SynthSeg uses: means  ~ U(0,250)   stds ~ U(0,35)
# ------------------------------------------------------------------
n_classes = GENERATION_CLASSES.max() + 1     # = 15 with the default label set

# "loc" = mid-point,  "scale" = half-range  (SampleConditionalGMMd convention)
mean_loc   = 125.0
mean_scale = 125.0                           # 0 … 250
std_loc    = 17.5
std_scale  = 17.5                            # 0 … 35

prior_means = np.vstack([
    np.full(n_classes, mean_loc,   dtype=float),
    np.full(n_classes, mean_scale, dtype=float),
])

prior_stds = np.vstack([
    np.full(n_classes, std_loc,    dtype=float),
    np.full(n_classes, std_scale,  dtype=float),
])

# probabilities required by the generator
prob = dict(
    flip        = 0.5,
    affine      = 0.9,
    contrast    = 0.5,
    gamma       = 0.5,
    scale_int   = 0.5,
    shift_int   = 0.5,
    hist_shift  = 0.5,
    noise       = 0.5,
    rician      = 0.3,
    gibbs       = 0.3,
    blur        = 0.3,
    bias        = 0.5,
    resolution  = 0.5,
)

brain_gen = BABrainGenerator(
    # Required parameters
    prior_means  = prior_means,
    prior_stds   = prior_stds,
    distribution = "uniform",           # ← uniform, no Gaussian priors
    prob         = prob,

    # Spatial augmentation parameters
    rotation_range     = 15,
    scaling_range      = 0.15,          # ±15 %
    shear_bounds       = 0.01,
    translation_bounds = 10,

    # Intensity augmentation parameters
    contrast_range      = (0.8, 1.2),
    log_gamma_std       = 0.2,
    shift_offset        = 0.1,
    hist_control_points = 5,

    # Artefacts parameters
    noise_mean    = 0.0,
    noise_std     = 0.03,
    rician_std    = 0.02,
    gibbs_alpha   = 0.4,
    blur_sigma    = 0.5,
    bias_field_rng= (0.0, 0.3),

    # Resolution parameters
    min_res       = 0.7,
    max_res_iso   = 3.0,
    max_res_aniso = 3.0,        # Default was 8.0
    atlas_res     = 1.0,        # Default was 1.0
    thickness     = 2.0,       # Default was None

    # SynthSeg label config parameters
    generation_labels = GENERATION_LABELS,   # Default was None (uses GENERATION_LABELS)
    n_neutral_labels  = N_NEUTRAL_LABELS,   # Default was None (uses N_NEUTRAL_LABELS)
    output_labels     = None,   # Default was None (no remapping)

    # Toggle parameters
    use_hemisphere_aware_flip     = True,  # Default was True
    use_dynamic_resolution        = True,  # Default was True
    use_intensity_clip_normalize  = False,  # Default was True
    n_channels                    = 1,     # Default was 1
    use_specific_stats_for_channel= False, # Default was False
    output_shape = (208, 240, 256),  # Should match the spatial dims (D, H, W)
    use_random_cropping          = True,   # Disable for debugging
    return_gradients             = False, # Default was False
)

# ------------------------------------------------------------------
# 3) OUTPUT FOLDER
# ------------------------------------------------------------------
out_dir = Path("C:/Projects/thesis_project/brain_age_pred/tests/brain_gen_images")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 4) GENERATE & SAVE 5 VOLUMES
# ------------------------------------------------------------------
for k in range(20):
    out_dict = brain_gen({"image": seg_data})     # forward pass
    vol = out_dict["image"].squeeze(0).cpu().numpy()    # Move to CPU for saving
    
    nifti = nib.Nifti1Image(vol, affine=seg_img.affine, header=seg_img.header)
    fname = out_dir / f"synthetic_{k:02d}.nii.gz"
    nib.save(nifti, fname)
    print(f"saved → {fname}")

print("Done.  Five volumes are in:", out_dir.resolve())