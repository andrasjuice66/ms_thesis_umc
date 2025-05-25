#!/usr/bin/env python
import numpy as np
import nibabel as nib

# Hardcoded paths
input_npy_path = "C:/Projects/thesis_project/Data/brain_age_converted/OASIS/OASIS3/sub-OAS30001_ses-d0129_T2w.nii.npy"
output_nifti_path = "C:/Projects/thesis_project/brain_age_pred/data/preprocess/output.nii.gz"

# Load the numpy array
data = np.load(input_npy_path)

# Create a NIfTI image
# Using identity matrix for affine since we don't have original header information
affine = np.eye(4)
nifti_img = nib.Nifti1Image(data, affine)

# Save the NIfTI file
nib.save(nifti_img, output_nifti_path)

print(f"Converted {input_npy_path} to {output_nifti_path}")
