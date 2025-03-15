import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Monai / TorchI0 (install if not available):
# pip install monai
# pip install torchio

import monai
import torchio as tio
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    RandAffine,
    RandGaussianNoise,
    RandBiasField,
    ScaleIntensity,
    Resize,
    Compose
)
from monai.data import MetaTensor

def plot_axial_slices(volume_list, titles, slice_idx=None):
    """
    Plots a list of 3D volumes in axial view on the same row.
    volume_list: list of numpy arrays [C, H, W, D] or [H, W, D]
    titles: list of string titles
    slice_idx: the index of axial slice to show (if None, picks the middle slice)
    """
    n = len(volume_list)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]  # Make it iterable

    for i, vol in enumerate(volume_list):
        # If it's 4D [C, H, W, D], assume single channel
        if len(vol.shape) == 4:
            vol = vol[0]  # remove channel dimension

        # Determine slice index
        depth = vol.shape[-1]
        if slice_idx is None:
            mid_slice = depth // 2
        else:
            mid_slice = slice_idx if slice_idx < depth else depth - 1

        axes[i].imshow(vol[..., mid_slice], cmap="gray", origin="lower")
        axes[i].axis("off")
        axes[i].set_title(titles[i], fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_axial_slices_with_diff(volume_list, titles, slice_idx=None):
    """
    Plots a list of 3D volumes in axial view with difference maps.
    volume_list: list of numpy arrays [C, H, W, D] or [H, W, D]
    titles: list of string titles
    slice_idx: the index of axial slice to show
    """
    n = len(volume_list)
    # Create figure with two rows: original images and difference maps
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    
    if n == 1:
        axes = [axes]  # Make it iterable
    
    for i, vol in enumerate(volume_list):
        # If it's 4D [C, H, W, D], assume single channel
        if len(vol.shape) == 4:
            vol = vol[0]  # remove channel dimension
            
        # Determine slice index
        depth = vol.shape[-1]
        if slice_idx is None:
            mid_slice = depth // 2
        else:
            mid_slice = slice_idx if slice_idx < depth else depth - 1
            
        # Plot original image
        axes[0, i].imshow(vol[..., mid_slice], cmap="gray", origin="lower")
        axes[0, i].axis("off")
        axes[0, i].set_title(titles[i], fontsize=10)
        
        # Plot difference map (skip for first image)
        if i > 0:
            prev_vol = volume_list[i-1]
            if len(prev_vol.shape) == 4:
                prev_vol = prev_vol[0]
            
            diff = vol[..., mid_slice] - prev_vol[..., mid_slice]
            # Normalize difference to [-1, 1] for visualization
            diff = diff / (np.abs(diff).max() + 1e-8)
            
            axes[1, i].imshow(diff, cmap="RdBu", origin="lower", vmin=-1, vmax=1)
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Difference Map\n({titles[i]} - {titles[i-1]})", fontsize=10)
        else:
            axes[1, i].axis("off")
            
    plt.tight_layout()
    plt.show()

def main(nifti_path):
    # 1) Load the image
    img_nii = nib.load(nifti_path)
    img_data = img_nii.get_fdata().astype(np.float32)
    # Store the original affine for saving later
    original_affine = img_nii.affine
    
    # MONAI typically expects shape [C, H, W, D], so we add a channel axis if needed
    if len(img_data.shape) == 3:
        img_data = np.expand_dims(img_data, axis=0)

    # Convert to MetaTensor (MONAI's standard format)
    img_tensor = MetaTensor(img_data, meta={"filename_or_obj": nifti_path})

    # Let's keep track of transformations step by step
    step_images = []
    step_titles = []

    # Determine the slice index once and use it consistently
    depth = img_data.shape[-1]
    slice_idx = depth // 2

    # Original image
    step_images.append(img_tensor.numpy())
    step_titles.append("Original")

    # 2) Metadata-based domain randomization transformations
    #    We will apply them step-by-step to visualize the effect.

    # (a) RandAffine (small random affine transformations)
    rand_affine = RandAffine(
        prob=1.0,  # always apply
        rotate_range=(0.1, 0.1, 0.1),     # increased rotation (in radians)
        shear_range=(0.1, 0.1),         # increased shear
        translate_range=(8, 8, 0),       # increased translation (but still no z-axis)
        scale_range=(0.1, 0.1, 0.1),   # increased scale changes
        mode="bilinear",
        padding_mode="border"
    )
    affined_tensor = rand_affine(img_tensor)
    step_images.append(affined_tensor.numpy())
    step_titles.append("Affine Transformed")

    # (b) Non-linear deformation using TorchIO
    elastic_transform = tio.RandomElasticDeformation(
        num_control_points=7,
        max_displacement=4,                # increased displacement
        locked_borders=2,
        include=['image']
    )
    # Note: TorchIO transforms expect a 'Subject'
    subject = tio.Subject(image=tio.ScalarImage(tensor=affined_tensor))
    warped_subject = elastic_transform(subject)
    warped_tensor = warped_subject.image.data
    step_images.append(warped_tensor.numpy())
    step_titles.append("Non-linear Warp")

    # (c) Bias Field (simulate scanner inhomogeneities)
    bias_transform = RandBiasField(prob=1.0, coeff_range=(0.3, 0.6))
    biased_tensor = bias_transform(MetaTensor(warped_tensor))
    step_images.append(biased_tensor.numpy())
    step_titles.append("Bias Field Applied")

    # (d) Random Noise injection (Gaussian noise)
    noise_transform = RandGaussianNoise(prob=1.0, mean=0.0, std=0.1)
    noisy_tensor = noise_transform(biased_tensor)
    step_images.append(noisy_tensor.numpy())
    step_titles.append("Gaussian Noise Added")

    # (e) Downsampling (simulate lower resolution) => monai.Resize or TorchIO Resample
    #     Here let's do a small resize to 0.5 in-plane
    original_shape = noisy_tensor.shape  # [C, H, W, D]
    downscale_factor = 0.5
    new_size = [int(original_shape[1] * downscale_factor),
                int(original_shape[2] * downscale_factor),
                int(original_shape[3] * downscale_factor)]
    resize_transform = Resize(spatial_size=new_size, mode="area")
    downsampled_tensor = resize_transform(noisy_tensor)

    # Optionally, you could re-upscale to original shape => show blocky artifacts
    # Let's do that to mimic typical "downsample then upsample" pipeline
    # i.e., first downsample, then upsample
    resize_transform_up = Resize(spatial_size=original_shape[1:], mode="area")
    upsampled_tensor = resize_transform_up(downsampled_tensor)

    step_images.append(upsampled_tensor.numpy())
    step_titles.append("Downsample")

    # After all transformations, save the final transformed image as NIFTI
    output_path = os.path.splitext(nifti_path)[0] + "_transformed.nii.gz"
    final_img = upsampled_tensor.numpy()[0]  # Remove channel dimension
    final_nifti = nib.Nifti1Image(final_img, original_affine)
    nib.save(final_nifti, output_path)

    # 3) Plot the axial slice progression with consistent slice_idx and difference maps
    plot_axial_slices_with_diff(step_images, step_titles, slice_idx=slice_idx)

if __name__ == "__main__":
    nifti_file = "/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/DataExp/data_transform/sub-01_T1w.nii.gz"
    main(nifti_file)

