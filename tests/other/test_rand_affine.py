import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import LoadImage, EnsureChannelFirst, RandAffine

# Path to your NIfTI image
IMG_PATH = r"C:\Projects\thesis_project\Data\brain_age_preprocessed\NIMH_RV\sub-ON02693_ses-01_acq-MPRAGE_T1w.nii.gz"

# Number of augmentations to generate
N_AUGS = 8

def load_volume(path):
    loader = LoadImage(image_only=True)
    ensure_channel_first = EnsureChannelFirst()
    img = loader(path)
    img = ensure_channel_first(img)
    return img  # shape: (1, Z, Y, X)

def plot_affine_augs(orig, aug_list, out_path=None):
    n = len(aug_list)
    ncols = 4
    nrows = (n + 1 + ncols - 1) // ncols  # +1 for original

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()

    # Show original
    orig_np = orig.squeeze().cpu().numpy()
    mid_slice = orig_np.shape[0] // 2  # axial
    axes[0].imshow(orig_np[mid_slice, :, :].T, cmap='gray', origin='lower')
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Show augmentations
    for i, aug in enumerate(aug_list):
        aug_np = aug.squeeze().cpu().numpy()
        axes[i+1].imshow(aug_np[mid_slice, :, :].T, cmap='gray', origin='lower')
        axes[i+1].set_title(f"RandAffine {i+1}")
        axes[i+1].axis("off")

    # Hide any unused subplots
    for j in range(n+1, nrows*ncols):
        axes[j].axis("off")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()
    plt.close()

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    img = load_volume(IMG_PATH)
    rand_affine = RandAffine(
        prob=1.0,
        rotate_range=(0.5, 0.5, 0.5),
        shear_range=(0.4, 0.1, 0.1),
        translate_range=(10, 10, 10),
        scale_range=(0.1, 0.1, 0.1),
        mode='bilinear'
    )

    aug_list = []
    for _ in range(N_AUGS):
        aug = rand_affine(img)
        aug_list.append(aug)

    plot_affine_augs(img, aug_list, out_path="brain_age_pred/tests/rand_affine_augs.png")

if __name__ == "__main__":
    main()
