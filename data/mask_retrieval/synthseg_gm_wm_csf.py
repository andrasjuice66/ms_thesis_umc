#!/usr/bin/env python3
"""
synthseg_tissue_masks.py
-----------------------------------------------
Run SynthSeg on an input NIfTI volume and
export GM / WM / CSF binary masks.
"""

import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def run_synthseg(in_img: Path, out_dir: Path, modality: str, threads: int = 4) -> Path:
    """
    Launch SynthSeg through Docker.
    The function returns the path to the generated segmentation.
    """
    seg_path = out_dir / f"seg_{modality}.nii.gz"
    vol_path = out_dir / f"volumes_{modality}.csv"  # Add path for volume measurements

    # Convert paths to Docker-compatible format
    in_dir = in_img.parent
    out_dir_abs = out_dir.absolute()
    in_dir_abs = in_dir.absolute()

    cmd = [
        "docker", "run", "--rm", "-it",
        "-v", f"{in_dir_abs}:/input",
        "-v", f"{out_dir_abs}:/output",
        "cookpa/synthseg:conda-0.1",
        "--i", f"/input/{in_img.name}",
        "--o", f"/output/seg_{modality}.nii.gz",
        "--vol", f"/output/volumes_{modality}.csv",  # Specify the output path for volume measurements
        "--robust",         # robust intensity normalisation
        "--threads", str(threads)
    ]
    print("[SynthSeg] " + " ".join(cmd))
    subprocess.check_call(cmd)

    return seg_path


def make_mask(segmentation: Path, labels: list[int], out_path: Path) -> None:
    """
    Build a binary mask where voxels with IDs in `labels` are 1.
    """
    seg_img = nib.load(segmentation)
    data    = seg_img.get_fdata()
    mask    = np.isin(data, labels).astype(np.uint8)

    nib.save(nib.Nifti1Image(mask, seg_img.affine, seg_img.header), out_path)
    print(f"  • saved {out_path.name}")


GM_LABELS = [   # cortical + subcortical grey matter
    3, 42,                       # cerebral cortex   L/R
    8, 47,                       # cerebellar cortex L/R
    9, 10, 11, 12, 13, 16,       # thalamus, caudate, putamen, pallidum, brain-stem
    18, 19, 20, 21, 22, 26, 27,  # hippocampus, amygdala, accumbens, etc.
    52, 53, 54                   # mirror-side equivalents
]
WM_LABELS = [
    2, 41,   # cerebral WM L/R
    7, 46    # cerebellar WM L/R
]
CSF_LABELS = [
    4, 43,               # lateral ventricles L/R
    14, 15, 24, 44, 72   # 3rd/4th ventricles, CSF, choroid plexus
]
def main() -> None:
    # Hardcoded paths
    young_image = Path("C:/Projects/thesis_project/Data/brain_age_preprocessed/OpenNeuro/BoldVariability/sub-146_T1w.nii.gz") 
    middle_image = Path("C:/Projects/thesis_project/Data/brain_age_preprocessed/IXI/IXI048-HH-1326-T1.nii.gz") 
    old_image = Path("C:/Projects/thesis_project/Data/brain_age_preprocessed/IXI/IXI252-HH-1693-T1.nii.gz")

    images = [young_image, middle_image, old_image]
    modalities = ["1_40", "40_60", "60_85"]  # List of modalities in same order as images
    out_dir = Path("C:/Projects/thesis_project/brain_age_pred/data/templates")  # Changed to templates directory
    threads = 4

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. segmentation ------------------------------------------------------
    for image, modality in zip(images, modalities):
        seg_path = run_synthseg(image, out_dir, modality, threads=threads)

        print(f"[Masks] generating GM / WM / CSF for {modality}...")
        # Changed output filenames to match simulation requirements
        make_mask(seg_path, GM_LABELS,  out_dir / f"{modality}_GM.nii.gz")
        make_mask(seg_path, WM_LABELS,  out_dir / f"{modality}_WM.nii.gz")
        make_mask(seg_path, CSF_LABELS, out_dir / f"{modality}_CSF.nii.gz")

        print(f"\nDone!  Results for {modality} are in:", out_dir)


if __name__ == "__main__":
    main()