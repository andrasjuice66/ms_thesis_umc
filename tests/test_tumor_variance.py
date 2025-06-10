#!/usr/bin/env python
"""
Visual demo for TumorSimulationModule - Multiple Runs for Variance Visualization
-------------------------------------------------------------------------------

• Requires: torch, numpy, nibabel, matplotlib, and the module file
  `tumor_simulation_complete.py` in your PYTHONPATH / same folder.
• Works on CPU; will use CUDA automatically if available.
• Runs tumor simulation 3 times to visualize variance
"""

import os, sys, random, pathlib, math
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from brain_age_pred.dom_rand.tumor_simulation import TumorSimulationModule

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# a) Where to find an example brain image (nifti or .npy) – leave None to synthesize one
BRAIN_IMG_PATH = r"C:/Projects/thesis_project/brain_age_pred/UNA/output/T1_original.nii.gz"

# b) (Optional) – provide age-specific segmentations if you have them
USE_SEG = True
SEGMENTATION_PATHS = {
    "young":  r"C:/Projects/thesis_project/brain_age_pred/data/segmentations/seg_18_40.nii.gz",
    "middle": r"C:/Projects/thesis_project/brain_age_pred/data/segmentations/seg_40_60.nii.gz",
    "old":    r"C:/Projects/thesis_project/brain_age_pred/data/segmentations/seg_60_85.nii.gz",
}
AGE_RANGES = {
    "young":  (18, 40),
    "middle": (40, 60),
    "old":    (60, 85),
}

# c) Tumour-sim parameters – tweak at will
sim_config = {
    "prob": 1.0,                             # always insert a tumour
    "modality": "T2",
    "perlin_res": [2, 2, 2],
    "use_age_based_segmentation": USE_SEG,
    "segmentation_paths": SEGMENTATION_PATHS if USE_SEG else None,
    "age_ranges": AGE_RANGES if USE_SEG else None,
    "use_fluid_dynamics": False,
}

# Number of simulations to run
NUM_SIMULATIONS = 3

# ---------------------------------------------------------
# 2. LOAD OR SYNTHESISE A 3-D IMAGE
# ---------------------------------------------------------
def load_or_make_image(path: str | None, shape=(128, 128, 128)) -> torch.Tensor:
    if path and pathlib.Path(path).exists():
        print(f"Loading image from {path}")
        if str(path).endswith(".npy"):
            arr = np.load(path)
        else:                               # assume NIfTI
            arr = nib.load(path).get_fdata()
        arr = arr.astype(np.float32)
    else:
        print("No valid image path given – generating synthetic brain volume.")
        # simple Gaussian blob + noise as a fake "brain"
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        center = np.array(shape)[:, None, None, None] / 2
        dist2 = ((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
        arr = np.exp(-dist2 / (2 * (shape[0] / 4) ** 2))          # ball-like intensity
        arr += 0.03 * np.random.randn(*shape)               # slight noise
        arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)   # rescale 0-1
    return torch.from_numpy(arr)

image_3d = load_or_make_image(BRAIN_IMG_PATH).to(device)     # [Z,Y,X]
image_3d = image_3d.unsqueeze(0)                             # add channel dim  -> [1,Z,Y,X]

# ---------------------------------------------------------
# 3. BUILD THE SIMULATOR & SAMPLE DICT
# ---------------------------------------------------------
simulator = TumorSimulationModule(device=device, **sim_config)

sample = {
    "image": image_3d,                                     # mandatory
    "age": torch.tensor(55.0),                           # demo age
    "modality": "T1",                                         # will use sim_config default otherwise
}

# ---------------------------------------------------------
# 4. RUN MULTIPLE SIMULATIONS & COLLECT OUTPUT
# ---------------------------------------------------------
print(f"\nRunning {NUM_SIMULATIONS} tumor simulations...")

results = []
stats = []

for i in range(NUM_SIMULATIONS):
    print(f"Running simulation {i+1}/{NUM_SIMULATIONS}...")
    
    with torch.no_grad():
        out = simulator(sample)

    tumor_mask = out["tumor_mask"].float()           # same shape as image (with channel)
    tumor_prob = out["tumor_prob"]
    diseased = out["image"]                        # image + tumour

    # Store results
    results.append({
        "tumor_mask": tumor_mask,
        "tumor_prob": tumor_prob,
        "diseased": diseased
    })

    # Calculate stats for this run
    voxels = tumor_mask.sum().item()
    if voxels > 0:
        com = torch.nonzero(tumor_mask).float().mean(0)  # centre of mass (c,z,y,x)
        zc, yc, xc = [int(v) for v in com[-3:].cpu()]
        
        # bounding box
        coords = torch.nonzero(tumor_mask[0])
        zmin, ymin, xmin = coords.min(0)[0].tolist()
        zmax, ymax, xmax = coords.max(0)[0].tolist()
        
        stats.append({
            "run": i + 1,
            "voxels": voxels,
            "center_of_mass": (zc, yc, xc),
            "bounding_box": ((zmin, zmax), (ymin, ymax), (xmin, xmax))
        })
    else:
        stats.append({
            "run": i + 1,
            "voxels": 0,
            "center_of_mass": None,
            "bounding_box": None
        })

# ---------------------------------------------------------
# 5. PRINT STATS FOR ALL RUNS
# ---------------------------------------------------------
print("\n" + "="*50)
print("TUMOUR STATISTICS ACROSS RUNS")
print("="*50)

for stat in stats:
    print(f"\nRun {stat['run']}:")
    print(f"  Voxel count            : {stat['voxels']}")
    print(f"  Approx. volume (mm³)   : {stat['voxels']:.1f}  (assumes 1 mm isotropic)")
    if stat['center_of_mass']:
        print(f"  Centre of mass (z,y,x) : {stat['center_of_mass']}")
        bb = stat['bounding_box']
        print(f"  Bounding box           : z[{bb[0][0]}:{bb[0][1]}], y[{bb[1][0]}:{bb[1][1]}], x[{bb[2][0]}:{bb[2][1]}]")
    else:
        print("  No tumour voxels detected")

# Calculate variance metrics
voxel_counts = [s['voxels'] for s in stats]
print(f"\nVariance Analysis:")
print(f"  Mean voxel count       : {np.mean(voxel_counts):.1f}")
print(f"  Std deviation          : {np.std(voxel_counts):.1f}")
print(f"  Min/Max voxel count    : {min(voxel_counts)}/{max(voxel_counts)}")

print("="*50)

# ---------------------------------------------------------
# 6. VISUALISATION - ALL RUNS IN ONE PLOT
# ---------------------------------------------------------
def show_slice(img, mask=None, title="", cmap="gray", axis=0, slice_idx=None):
    data = img.squeeze().cpu().numpy()
    if slice_idx is None:
        slice_idx = data.shape[axis] // 2
    slc = np.take(data, slice_idx, axis=axis)

    plt.imshow(slc.T, cmap=cmap, origin="lower")
    if mask is not None:
        mdata = mask.squeeze().cpu().numpy()
        mslc = np.take(mdata, slice_idx, axis=axis)
        plt.imshow(mslc.T, cmap="Reds", alpha=0.35, origin="lower")
    plt.title(title)
    plt.axis("off")

# Create a large figure to show all results
fig = plt.figure(figsize=(18, 12))

# First row: Original image (repeated for comparison)
plt.subplot(4, NUM_SIMULATIONS, 1)
show_slice(image_3d, None, "Original Brain")
for i in range(1, NUM_SIMULATIONS):
    plt.subplot(4, NUM_SIMULATIONS, i + 1)
    show_slice(image_3d, None, "Original Brain")

# Second row: Diseased images with tumor overlays
for i in range(NUM_SIMULATIONS):
    plt.subplot(4, NUM_SIMULATIONS, NUM_SIMULATIONS + i + 1)
    show_slice(results[i]["diseased"], results[i]["tumor_mask"], 
               f"Run {i+1}: Tumor Overlay\n({stats[i]['voxels']} voxels)")

# Third row: Tumor masks only
for i in range(NUM_SIMULATIONS):
    plt.subplot(4, NUM_SIMULATIONS, 2 * NUM_SIMULATIONS + i + 1)
    show_slice(results[i]["tumor_mask"], None, f"Run {i+1}: Tumor Mask", cmap="Reds")

# Fourth row: Tumor probability maps
for i in range(NUM_SIMULATIONS):
    plt.subplot(4, NUM_SIMULATIONS, 3 * NUM_SIMULATIONS + i + 1)
    show_slice(results[i]["tumor_prob"], None, f"Run {i+1}: Tumor Probability", cmap="hot")

plt.tight_layout()
plt.suptitle(f"Tumor Simulation Variance Analysis ({NUM_SIMULATIONS} Runs)", 
             fontsize=16, y=0.98)
plt.show()

# ---------------------------------------------------------
# 7. SAVE ALL OUTPUTS AS NIFTI
# ---------------------------------------------------------
out_dir = pathlib.Path("output_nifti_variance")
out_dir.mkdir(exist_ok=True)

# Use the source file name (or "synthetic") as a prefix
prefix = (
    pathlib.Path(BRAIN_IMG_PATH).stem
    if BRAIN_IMG_PATH and pathlib.Path(BRAIN_IMG_PATH).exists()
    else "synthetic"
)

# Get an affine / header to preserve spatial info
if BRAIN_IMG_PATH and pathlib.Path(BRAIN_IMG_PATH).exists():
    src_nii = nib.load(BRAIN_IMG_PATH)
    affine = src_nii.affine
    header = src_nii.header.copy()
else:
    affine = np.eye(4, dtype=np.float32)           # identity if synthetic
    header = None

print(f"\nSaving NIfTI files for all {NUM_SIMULATIONS} runs...")

saved_files = []
for i, result in enumerate(results):
    # Convert tensors → NumPy and drop the channel dim
    diseased_np = result["diseased"].squeeze(0).cpu().numpy().astype(np.float32)
    mask_np = result["tumor_mask"].squeeze(0).cpu().numpy().astype(np.uint8)
    prob_np = result["tumor_prob"].squeeze(0).cpu().numpy().astype(np.float32)

    # File names with run number
    diseased_file = out_dir / f"{prefix}_run{i+1}_with_tumour.nii.gz"
    mask_file = out_dir / f"{prefix}_run{i+1}_tumour_mask.nii.gz"
    prob_file = out_dir / f"{prefix}_run{i+1}_tumour_prob.nii.gz"

    # Save files
    nib.save(nib.Nifti1Image(diseased_np, affine, header), diseased_file)
    nib.save(nib.Nifti1Image(mask_np, affine, header), mask_file)
    nib.save(nib.Nifti1Image(prob_np, affine, header), prob_file)
    
    saved_files.extend([diseased_file, mask_file, prob_file])

print(f"\nSaved {len(saved_files)} NIfTI files to {out_dir}/")
for file in saved_files:
    print(f"  {file.name}") 