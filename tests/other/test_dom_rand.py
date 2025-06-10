"""
Domain-Randomizer showcase
=========================

Run this file (`python test_domain_randomizer.py`) and it will:

1. Load the reference T1 volume (hard-coded path below)
2. Loop over every transform supported by `DomainRandomizer`
3. Turn that transform **on** (p=1) and every other one **off** (p=0)
4. Apply the augmentation once
5. Write a PNG of the central axial slice to:

   <script_dir>/dr_examples/<transform_name>.png
"""

from __future__ import annotations
import os, copy
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch

# ------------------------------------------------------------------ #
#  EDIT HERE if your project import path is different
# ------------------------------------------------------------------ #
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer

# ------------------------------------------------------------------ #
#  HARD-CODED INPUT VOLUME (edit if you move the file)
# ------------------------------------------------------------------ #
IMG_PATH = Path(r"C:\Projects\thesis_project\Data\brain_age_preprocessed\CamCAN\sub-CC420148_T1w.nii.gz")

# ------------------------------------------------------------------ #
#  OUTPUT FOLDER (created next to this script)
# ------------------------------------------------------------------ #
OUT_DIR = Path(__file__).with_suffix("").parent / "dr_examples"
OUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------ #
#                       helper: save png                             #
# ------------------------------------------------------------------ #
def save_png(volume: np.ndarray, fname: Path, idx: int | None = None) -> None:
    if idx is None:
        idx = volume.shape[2] // 2
    plt.figure(figsize=(4, 4))
    plt.imshow(volume[:, :, idx].T, cmap="gray", origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ------------------------------------------------------------------ #
def main() -> None:
    print(f"✓ Loading volume: {IMG_PATH}")
    nii   = nib.load(str(IMG_PATH))
    data  = nii.get_fdata(dtype=np.float32)
    data  = (data - data.min()) / (data.ptp() + 1e-8)          # 0–1
    tensor = torch.from_numpy(data).unsqueeze(0)               # (1,D,H,W)

    for tname in DomainRandomizer._DEFAULT_PROBS.keys():
        print(f"▶  Applying: {tname:<10s}", end=" ... ")

        # enable *only* this transform
        probs = {k: 0.0 for k in DomainRandomizer._DEFAULT_PROBS}
        probs[tname] = 1.0

        dr = DomainRandomizer(
            device=torch.device("cpu"),
            transform_probs=probs,
            output_shape=None,      # skip random crop to keep size
            use_torchio=True,       # needed for elastic / ghost / spike
        )

        out_sample = dr({"image": tensor.clone()})
        out_vol    = out_sample["image"].squeeze(0).numpy()

        save_png(out_vol, OUT_DIR / f"{tname}.png")
        print("saved.")

    print(f"\n✓ All PNGs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()