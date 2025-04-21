import numpy as np
import torch
import torchio as tio
import nibabel as nib
import matplotlib.pyplot as plt
import random
import os


#!/usr/bin/env python
# artifact_simulation.ipynb
#
# Example Notebook/Script using TorchIO to apply various MRI-like artifacts
# to a brain NIfTI image, then visualize the results alongside the original.

import numpy as np
import torchio as tio
import nibabel as nib
import matplotlib.pyplot as plt

# Helper function to extract a 2D slice from a 3D volume for plotting
def get_middle_slice(image_data):
    """Return an axial slice (top-down view) from a 3D image array."""
    if image_data.ndim == 4:  # TorchIO output: (channels, depth, height, width)
        c, d, h, w = image_data.shape
        # For an axial slice from top to bottom, fix the depth at d//2
        return image_data[0, d // 2, :, :]
    elif image_data.ndim == 3:  # Nibabel output: (depth, height, width)
        d, h, w = image_data.shape
        return image_data[d // 2, :, :]
    else:
        raise ValueError("Expected image data with 3 or 4 dimensions.")

# Create output directory if it doesn't exist
output_dir = '/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/DataExp/data_transform/augmented_images'
os.makedirs(output_dir, exist_ok=True)

# Load the original image
input_path = '/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/DataExp/data_transform/sub-01_T1w.nii.gz'
original_image = tio.ScalarImage(input_path)

# Get the base filename without extension
base_filename = os.path.splitext(os.path.basename(input_path))[0]
if base_filename.endswith('.nii'):  # Handle .nii.gz case
    base_filename = os.path.splitext(base_filename)[0]

# Define a list of TorchIO transforms that mimic different artifacts
transforms = [
    ("Motion", tio.RandomMotion(degrees=10, translation=5, num_transforms=1)),
    ("Ghosting", tio.RandomGhosting(intensity=0.7, num_ghosts=4)),
    ("Spike", tio.RandomSpike(num_spikes=1, intensity=(10, 15))),
    ("BiasField", tio.RandomBiasField(coefficients=0.5)),
    ("Blur", tio.RandomBlur(std=(1.0, 2.0))),
    ("Noise", tio.RandomNoise(mean=0, std=0.05)),
    # Added new artifacts
    #("Downsampling", tio.RandomDownsampling(downsampling=(2, 2.5))),
    ("Anisotropy", tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 2.5))),
    ("SwapAxis", tio.RandomSwap(patch_size=15, num_iterations=3)),
    # ("ElasticDeformation", tio.RandomElasticDeformation(max_displacement=7.5)),
    # ("GammaTransform", tio.RandomGamma(log_gamma=(-0.3, 0.3))),
    #("MotionFromResampling", tio.RandomMotionFromResampling(degrees=10, translation=5)),
]

# Create a figure to display all transformations
plt.figure(figsize=(20, 12))
n_transforms = len(transforms) + 1  # +1 for original image
rows = 3
cols = (n_transforms + 2) // 3  # Adjust number of columns to fit all images

# Plot original image first
original_slice = get_middle_slice(original_image.data.numpy())
plt.subplot(rows, cols, 1)
plt.imshow(original_slice, cmap='gray')
plt.title('Original')
plt.axis('off')

# Apply each transform, save full 3D image, and plot middle slice
for idx, (artifact_name, transform) in enumerate(transforms, start=1):
    # Apply the transform
    transformed = transform(original_image)
    
    # Save the full 3D transformed image
    output_filename = f"{base_filename}_{artifact_name.lower()}.nii.gz"
    output_path = os.path.join(output_dir, output_filename)
    transformed.save(output_path)
    print(f"Saved {artifact_name} transformed image to: {output_path}")
    
    # Plot middle slice for visualization
    transformed_slice = get_middle_slice(transformed.data.numpy())
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(transformed_slice, cmap='gray')
    plt.title(artifact_name)
    plt.axis('off')

# Save original image
original_output_path = os.path.join(output_dir, f"{base_filename}_original.nii.gz")
original_image.save(original_output_path)
print(f"Saved original image to: {original_output_path}")

# Adjust layout and display
plt.tight_layout()
plt.show()

