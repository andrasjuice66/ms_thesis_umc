import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from monai.transforms import (
    RandFlipd,
    RandAffined,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianSmoothd,
    RandSpatialCropd,
    ToTensord,
    LoadImaged,
    EnsureChannelFirstd,
)
from brain_age_pred.dom_rand.custom_transformations import (
    RandomResolutionD,
    RandGammaD,
)

def plot_transform_comparison(original, transformed, title, slice_idx=None):
    """Plot original and transformed images side by side."""
    if slice_idx is None:
        slice_idx = original.shape[-1] // 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original
    ax1.imshow(original[0, :, :, slice_idx], cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Plot transformed
    ax2.imshow(transformed[0, :, :, slice_idx], cmap='gray')
    ax2.set_title(f'After {title}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_transforms(image_path, device=torch.device("cpu")):
    """Visualize all transforms in the domain randomization pipeline."""
    # Load and preprocess image
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ])
    
    # Load the image
    sample = {"image": image_path}
    sample = transforms(sample)
    original_image = sample["image"]
    
    # Define transforms with their parameters
    transforms_to_visualize = {
        "Random Flip": RandFlipd(
            keys=["image"], 
            prob=1.0,  # Force transform for visualization
            spatial_axis=0
        ),
        "Random Affine": RandAffined(
            keys=["image"],
            prob=1.0,
            rotate_range=(np.pi/18, np.pi/18, np.pi/18),  # 10 degrees
            scale_range=(0.1, 0.1, 0.1),
            shear_range=(0.05, 0.05, 0.05),
            mode="bilinear",
        ),
        "Random Contrast": RandAdjustContrastd(
            keys=["image"],
            prob=1.0,
            gamma=(0.8, 1.2),
        ),
        "Random Gamma": RandGammaD(
            keys=["image"],
            prob=1.0,
            log_gamma_std=0.2,
        ),
        "Random Gaussian Smooth": RandGaussianSmoothd(
            keys=["image"],
            prob=1.0,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        "Random Bias Field": RandBiasFieldd(
            keys=["image"],
            prob=1.0,
            coeff_range=(0.0, 0.4),
        ),
        "Random Resolution": RandomResolutionD(
            keys=["image"],
            prob=1.0,
            min_res=1.0,
            max_res_iso=3.0,
        ),
    }
    
    # Apply and visualize each transform
    for transform_name, transform in transforms_to_visualize.items():
        # Apply transform
        transformed_sample = transform({"image": original_image})
        transformed_image = transformed_sample["image"]
        
        # Plot comparison
        plot_transform_comparison(
            original_image.numpy(),
            transformed_image.numpy(),
            transform_name
        )

if __name__ == "__main__":
    # Example usage
    image_path = "C:/Projects/thesis_project/Data/brain_age_preprocessed/OpenNeuro/DallasLifeSpan/sub-12_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz"  # Replace with your image path
    visualize_transforms(image_path)
