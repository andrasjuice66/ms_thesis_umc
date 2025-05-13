#!/usr/bin/env python
"""
Quick visual sanity-check for DomainRandomizer.

Usage
-----
python visualize_domain_randomizer.py            # GPU if available
python visualize_domain_randomizer.py --device cpu --n_aug 5
"""

from __future__ import annotations
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt

from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer


# ------------------------------------------------------------------------- #
#                             helper functions                              #
# ------------------------------------------------------------------------- #
def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI file and z-score it."""
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    data = (data - data.mean()) / (data.std() + 1e-8)  # simple z-score
    return data


def plot_three_planes(axs, volume: np.ndarray, title_prefix: str) -> None:
    """
    Show central axial, coronal, sagittal slices.
    `axs` must be an iterable of exactly three matplotlib axes.
    """
    D, H, W = volume.shape
    slices  = (
        volume[D // 2, :, :],   # axial
        volume[:, H // 2, :],   # coronal
        volume[:, :, W // 2],   # sagittal
    )
    for ax, slc, plane in zip(axs, slices, ("Axial", "Coronal", "Sagittal")):
        ax.imshow(slc.T, cmap="gray", origin="lower")
        ax.set_title(f"{title_prefix}\n{plane}")
        ax.axis("off")


def visualize_single_image(
    img_path: str | Path,
    dr: DomainRandomizer,
    n_aug: int = 3,
) -> None:
    """
    Plot original + `n_aug` randomly augmented versions of a single volume.
    """
    vol = load_nifti(img_path)

    # build torch sample once (on CPU – DR handles device internally)
    base_sample = {"image": torch.from_numpy(vol).unsqueeze(0)}

    # figure grid: 1 row original + n_aug rows augmented
    fig, axes = plt.subplots(n_aug + 1, 3, figsize=(11, 3 * (n_aug + 1)))

    # row 0 – original
    plot_three_planes(axes[0], vol, "Original")

    # rows 1..n – augmentations
    for i in range(n_aug):
        aug_sample = dr({"image": base_sample["image"].clone()})
        aug_vol    = aug_sample["image"].cpu().numpy()[0]
        plot_three_planes(axes[i + 1], aug_vol, f"Augmented #{i + 1}")

    plt.suptitle(Path(img_path).name)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------- #
#                                    main                                   #
# ------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Visualize DomainRandomizer output")
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Where to run MONAI transforms (default: auto-detect GPU).",
    )
    ap.add_argument(
        "--n_aug",
        type=int,
        default=3,
        help="How many random augmentations to visualise per image.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    # Instantiate with default probabilities / ranges
    dr = DomainRandomizer(device=device)

    img_paths = [
        r"C:\Projects\thesis_project\Data\brain_age_preprocessed\CamCAN\sub-CC110101_T2w.nii.gz",
        r"C:\Projects\thesis_project\Data\brain_age_preprocessed\CamCAN\sub-CC420148_T1w.nii.gz",
    ]

    for p in img_paths:
        visualize_single_image(p, dr, n_aug=args.n_aug)


if __name__ == "__main__":
    main()