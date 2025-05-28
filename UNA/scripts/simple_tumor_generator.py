#!/usr/bin/env python3
"""
Simple Tumor Generator using UNA Framework
==========================================

This script generates synthetic tumors on brain images using Perlin noise.
It's a simplified version that doesn't require the complex dataset setup.

Requirements:
- Input brain image (NIfTI format)
- Brain segmentation (NIfTI format) with labels: 0=background, 1=GM, 2=WM, 3=CSF
- Modality type (T1, T2, FLAIR)

Usage:
    python simple_tumor_generator.py --input brain.nii.gz --seg seg.nii.gz --modality T1 --output ./output
"""

import os
import sys
import argparse
import time
import datetime
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import nibabel as nib

import utils.misc as utils
from FluidAnomaly.perlin3d import generate_shape_3d, generate_velocity_3d
from FluidAnomaly.DiffEqs.pde import AdvDiffPDE
from FluidAnomaly.DiffEqs.odeint import odeint


class SimpleTumorGenerator:
    """Simplified tumor generator using Perlin noise"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Default parameters for tumor generation
        self.shape_gen_args = {
            'perlin_res': [2, 2, 2],
            'mask_percentile_min': 90,
            'mask_percentile_max': 99.6,
            'integ_method': 'dopri5',
            'bc': 'neumann',
            'V_multiplier': 500,
            'dt': 0.1,
            'min_nt': 10,
            'max_nt': 20,
            'pathol_thres': 0.2,
            'pathol_tol': 0.000001
        }
        
        # Initialize PDE for fluid dynamics
        self.t = torch.from_numpy(np.arange(self.shape_gen_args['max_nt']) * self.shape_gen_args['dt']).to(self.device)
        with torch.no_grad():
            self.adv_pde = AdvDiffPDE(
                data_spacing=[1., 1., 1.], 
                perf_pattern='adv', 
                V_type='vector_div_free', 
                V_dict={},
                BC=self.shape_gen_args['bc'], 
                dt=self.shape_gen_args['dt'], 
                device=self.device
            )
    
    def generate_tumor_shape(self, shape):
        """Generate tumor shape using Perlin noise"""
        percentile = np.random.uniform(
            self.shape_gen_args['mask_percentile_min'], 
            self.shape_gen_args['mask_percentile_max']
        )
        
        tumor_prob, tumor_mask = generate_shape_3d(
            shape, 
            self.shape_gen_args['perlin_res'], 
            percentile, 
            self.device
        )
        
        return tumor_prob, tumor_mask
    
    def augment_tumor_with_fluid_dynamics(self, tumor_prob):
        """Apply fluid dynamics to make tumor shape more realistic"""
        tumor_prob = torch.squeeze(tumor_prob)
        
        # Generate random number of time steps
        nt = np.random.randint(self.shape_gen_args['min_nt'], self.shape_gen_args['max_nt'] + 1)
        
        try:
            # Generate velocity field
            self.adv_pde.V_dict = generate_velocity_3d(
                tumor_prob.shape, 
                self.shape_gen_args['perlin_res'], 
                self.shape_gen_args['V_multiplier'], 
                self.device
            )
            
            # Apply PDE evolution
            tumor_prob = odeint(
                self.adv_pde, 
                tumor_prob[None], 
                self.t[:nt], 
                self.shape_gen_args['dt'], 
                method=self.shape_gen_args['integ_method']
            )[-1, 0]  # Take the last time step
            
        except Exception as e:
            print(f'Warning: Exception during PDE augmentation: {e}')
            # Return original if PDE fails
            pass
        
        return tumor_prob
    
    def get_contrast_values(self, modality):
        """Get contrast values for different tissue types based on modality"""
        if modality.upper() == 'T1':
            # T1-weighted: CSF dark, GM medium, WM bright
            mus = torch.tensor([0, 100, 150, 50], dtype=torch.float, device=self.device)  # [bg, GM, WM, CSF]
            sigmas = torch.tensor([0, 15, 20, 10], dtype=torch.float, device=self.device)
        elif modality.upper() == 'T2':
            # T2-weighted: CSF bright, GM medium, WM dark
            mus = torch.tensor([0, 120, 80, 200], dtype=torch.float, device=self.device)  # [bg, GM, WM, CSF]
            sigmas = torch.tensor([0, 18, 12, 25], dtype=torch.float, device=self.device)
        elif modality.upper() == 'FLAIR':
            # FLAIR: CSF dark, GM medium, WM bright, lesions bright
            mus = torch.tensor([0, 110, 140, 20], dtype=torch.float, device=self.device)  # [bg, GM, WM, CSF]
            sigmas = torch.tensor([0, 16, 18, 5], dtype=torch.float, device=self.device)
        else:
            raise ValueError(f"Unsupported modality: {modality}. Use T1, T2, or FLAIR")
        
        return mus, sigmas
    
    def encode_pathology(self, image, tumor_prob, modality):
        """Encode tumor pathology into the image"""
        # Calculate mean intensity in white matter (label 2)
        wm_mask = (image > 0)  # Simple mask for non-zero regions
        if wm_mask.sum() > 0:
            wm_mean = (image * wm_mask).sum() / wm_mask.sum()
        else:
            wm_mean = 100.0  # Default value
        
        # Determine pathology direction based on modality
        if modality.upper() in ['T2', 'FLAIR']:
            pathol_direction = True  # Hyperintense (brighter)
            intensity_multiplier = 1.5 + 0.5 * torch.rand(1, device=self.device)
        else:  # T1
            pathol_direction = False  # Hypointense (darker)
            intensity_multiplier = 0.3 + 0.4 * torch.rand(1, device=self.device)
        
        # Apply pathology
        tumor_mask = torch.round(tumor_prob).long()
        pathol_intensity = wm_mean * intensity_multiplier
        
        if pathol_direction:
            # Hyperintense lesion
            image += tumor_prob * pathol_intensity
        else:
            # Hypointense lesion
            image = image * (1 - tumor_prob * 0.7) + tumor_prob * pathol_intensity
        
        # Ensure non-negative values
        image[image < 0] = 0
        
        return image
    
    def generate_synthetic_image(self, segmentation, modality):
        """Generate synthetic brain image from segmentation"""
        seg_tensor = torch.from_numpy(segmentation).long().to(self.device)
        
        # Get contrast values
        mus, sigmas = self.get_contrast_values(modality)
        
        # Generate synthetic image
        synthetic_image = mus[seg_tensor] + sigmas[seg_tensor] * torch.randn(seg_tensor.shape, dtype=torch.float, device=self.device)
        
        # Ensure non-negative values
        synthetic_image[synthetic_image < 0] = 0
        
        return synthetic_image
    
    def generate_tumor_on_image(self, input_image, segmentation, modality, tumor_size_factor=1.0):
        """
        Generate a tumor on the input image
        
        Args:
            input_image: numpy array of the brain image
            segmentation: numpy array of brain segmentation (0=bg, 1=GM, 2=WM, 3=CSF)
            modality: string, one of 'T1', 'T2', 'FLAIR'
            tumor_size_factor: float, factor to control tumor size (default 1.0)
        
        Returns:
            dict with 'diseased_image', 'tumor_mask', 'tumor_prob'
        """
        print(f"Generating {modality} tumor...")
        
        # Convert to tensors
        if isinstance(input_image, np.ndarray):
            image_tensor = torch.from_numpy(input_image).float().to(self.device)
        else:
            image_tensor = input_image.clone()
        
        seg_tensor = torch.from_numpy(segmentation).long().to(self.device)
        
        # Generate tumor shape
        tumor_prob, tumor_mask = self.generate_tumor_shape(image_tensor.shape)
        
        # Apply fluid dynamics augmentation
        tumor_prob = self.augment_tumor_with_fluid_dynamics(tumor_prob)
        
        # Scale tumor size
        if tumor_size_factor != 1.0:
            tumor_prob = tumor_prob * tumor_size_factor
            tumor_prob = torch.clamp(tumor_prob, 0, 1)
        
        # Restrict tumor to brain tissue (GM and WM)
        brain_mask = (seg_tensor == 1) | (seg_tensor == 2)  # GM or WM
        tumor_prob = tumor_prob * brain_mask.float()
        
        # Check if tumor is large enough
        if tumor_prob.sum() < 100:  # Minimum tumor size
            print("Generated tumor too small, regenerating...")
            return self.generate_tumor_on_image(input_image, segmentation, modality, tumor_size_factor * 1.5)
        
        # Create final tumor mask
        tumor_mask = (tumor_prob > self.shape_gen_args['pathol_thres']).float()
        
        # Apply pathology to image
        diseased_image = image_tensor.clone()
        diseased_image = self.encode_pathology(diseased_image, tumor_prob, modality)
        
        # Normalize image
        if diseased_image.max() > 0:
            diseased_image = diseased_image / diseased_image.max() * image_tensor.max()
        
        return {
            'diseased_image': diseased_image.cpu().numpy(),
            'tumor_mask': tumor_mask.cpu().numpy(),
            'tumor_prob': tumor_prob.cpu().numpy()
        }


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic tumors using UNA framework')
    parser.add_argument('--input', type=str, required=True, help='Input brain image (NIfTI)')
    parser.add_argument('--seg', type=str, required=True, help='Brain segmentation (NIfTI)')
    parser.add_argument('--modality', type=str, choices=['T1', 'T2', 'FLAIR'], required=True, help='Image modality')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--tumor-size', type=float, default=1.0, help='Tumor size factor (default: 1.0)')
    parser.add_argument('--num-tumors', type=int, default=1, help='Number of tumors to generate (default: 1)')
    parser.add_argument('--use-synthetic', action='store_true', help='Generate synthetic image from segmentation instead of using input')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.exists(args.seg):
        raise FileNotFoundError(f"Segmentation file not found: {args.seg}")
    
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images
    print("Loading images...")
    input_img = nib.load(args.input)
    input_data = input_img.get_fdata()
    affine = input_img.affine
    
    seg_img = nib.load(args.seg)
    seg_data = seg_img.get_fdata()
    
    # Validate segmentation labels
    unique_labels = np.unique(seg_data)
    expected_labels = np.array([0, 1, 2, 3])
    
    print(f"Image shape: {input_data.shape}")
    print(f"Segmentation labels found: {unique_labels}")
    print(f"Expected labels: {expected_labels}")
    
    # Check if all expected labels are present
    missing_labels = set(expected_labels) - set(unique_labels)
    extra_labels = set(unique_labels) - set(expected_labels)
    
    if missing_labels:
        print(f"WARNING: Missing expected labels: {missing_labels}")
    if extra_labels:
        print(f"WARNING: Unexpected labels found: {extra_labels}")
    
    # Check if segmentation has the basic required labels (GM=1, WM=2)
    if 1 not in unique_labels or 2 not in unique_labels:
        raise ValueError("Segmentation must contain at least GM (label=1) and WM (label=2) for tumor generation")
    
    # Print label statistics
    for label in unique_labels:
        count = np.sum(seg_data == label)
        percentage = (count / seg_data.size) * 100
        label_name = {0: 'Background', 1: 'Gray Matter', 2: 'White Matter', 3: 'CSF'}.get(int(label), f'Unknown({int(label)})')
        print(f"  Label {int(label)} ({label_name}): {count:,} voxels ({percentage:.1f}%)")
    
    print(f"Segmentation labels: {np.unique(seg_data)}")
    
    # Initialize generator
    generator = SimpleTumorGenerator(device=device)
    
    # Generate tumors
    start_time = time.time()
    
    for i in range(args.num_tumors):
        print(f"\nGenerating tumor {i+1}/{args.num_tumors}...")
        
        # Use synthetic image if requested
        if args.use_synthetic:
            print("Generating synthetic brain image from segmentation...")
            input_for_tumor = generator.generate_synthetic_image(seg_data, args.modality).cpu().numpy()
        else:
            input_for_tumor = input_data.copy()
        
        # Generate tumor
        result = generator.generate_tumor_on_image(
            input_for_tumor, 
            seg_data, 
            args.modality, 
            args.tumor_size
        )
        
        # Save results
        suffix = f"_{i+1}" if args.num_tumors > 1 else ""
        
        # Save diseased image
        diseased_img = nib.Nifti1Image(result['diseased_image'], affine)
        diseased_path = output_dir / f"{args.modality}_with_tumor{suffix}.nii.gz"
        nib.save(diseased_img, diseased_path)
        print(f"Saved diseased image: {diseased_path}")
        
        # Save tumor mask
        tumor_mask_img = nib.Nifti1Image(result['tumor_mask'], affine)
        tumor_mask_path = output_dir / f"tumor_mask{suffix}.nii.gz"
        nib.save(tumor_mask_img, tumor_mask_path)
        print(f"Saved tumor mask: {tumor_mask_path}")
        
        # Save tumor probability
        tumor_prob_img = nib.Nifti1Image(result['tumor_prob'], affine)
        tumor_prob_path = output_dir / f"tumor_probability{suffix}.nii.gz"
        nib.save(tumor_prob_img, tumor_prob_path)
        print(f"Saved tumor probability: {tumor_prob_path}")
        
        # Save original for comparison (only once)
        if i == 0:
            if args.use_synthetic:
                orig_img = nib.Nifti1Image(input_for_tumor, affine)
                orig_path = output_dir / f"{args.modality}_synthetic_original.nii.gz"
            else:
                orig_img = nib.Nifti1Image(input_data, affine)
                orig_path = output_dir / f"{args.modality}_original.nii.gz"
            nib.save(orig_img, orig_path)
            print(f"Saved original image: {orig_path}")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nGeneration completed in {total_time_str}')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main() 