#!/usr/bin/env python3
"""
Example Usage of Simple Tumor Generator
=======================================

This script demonstrates how to use the SimpleTumorGenerator class
to create synthetic tumors on brain images.
"""

import os
import sys
import numpy as np
import nibabel as nib

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.simple_tumor_generator import SimpleTumorGenerator


def create_example_segmentation(shape=(128, 128, 128)):
    """Create a simple example brain segmentation for testing"""
    seg = np.zeros(shape, dtype=np.uint8)
    
    # Create a simple brain-like structure
    center = np.array(shape) // 2
    
    # Create spherical regions
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Brain mask (everything inside a large sphere)
    brain_radius = min(shape) // 3
    brain_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < brain_radius**2
    
    # White matter (inner sphere)
    wm_radius = brain_radius * 0.6
    wm_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < wm_radius**2
    
    # CSF (very center)
    csf_radius = brain_radius * 0.2
    csf_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < csf_radius**2
    
    # Assign labels
    seg[brain_mask] = 1  # Gray matter
    seg[wm_mask] = 2     # White matter
    seg[csf_mask] = 3    # CSF
    
    return seg


def create_example_brain_image(segmentation, modality='T1'):
    """Create a simple example brain image from segmentation"""
    # Simple intensity mapping based on modality
    if modality.upper() == 'T1':
        intensity_map = {0: 0, 1: 100, 2: 150, 3: 50}  # CSF dark, WM bright
    elif modality.upper() == 'T2':
        intensity_map = {0: 0, 1: 120, 2: 80, 3: 200}  # CSF bright, WM dark
    elif modality.upper() == 'FLAIR':
        intensity_map = {0: 0, 1: 110, 2: 140, 3: 20}  # CSF dark, WM bright
    else:
        intensity_map = {0: 0, 1: 100, 2: 150, 3: 50}
    
    # Create image
    image = np.zeros_like(segmentation, dtype=np.float32)
    for label, intensity in intensity_map.items():
        image[segmentation == label] = intensity
    
    # Add some noise for realism
    noise = np.random.normal(0, 5, image.shape)
    image = image + noise
    image[image < 0] = 0
    
    return image


def example_with_real_data():
    """Example using real brain data (if available)"""
    # Example paths - modify these to point to your actual data
    input_path = "path/to/your/brain.nii.gz"
    seg_path = "path/to/your/segmentation.nii.gz"
    
    if not os.path.exists(input_path) or not os.path.exists(seg_path):
        print("Real data not found, skipping real data example")
        return
    
    print("=== Example with Real Data ===")
    
    # Load data
    input_img = nib.load(input_path)
    input_data = input_img.get_fdata()
    
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    
    # Initialize generator
    generator = SimpleTumorGenerator(device='cpu')  # Use 'cuda' if available
    
    # Generate tumor
    result = generator.generate_tumor_on_image(
        input_data, 
        seg_data, 
        modality='T1',
        tumor_size_factor=1.0
    )
    
    # Save results
    output_dir = "output_real_data"
    os.makedirs(output_dir, exist_ok=True)
    
    nib.save(nib.Nifti1Image(result['diseased_image'], input_img.affine), 
             f"{output_dir}/T1_with_tumor.nii.gz")
    nib.save(nib.Nifti1Image(result['tumor_mask'], input_img.affine), 
             f"{output_dir}/tumor_mask.nii.gz")
    
    print(f"Results saved to {output_dir}/")


def example_with_synthetic_data():
    """Example using synthetic brain data"""
    print("=== Example with Synthetic Data ===")
    
    # Create synthetic data
    shape = (128, 128, 128)
    seg_data = create_example_segmentation(shape)
    
    # Initialize generator
    generator = SimpleTumorGenerator(device='cpu')  # Use 'cuda' if available
    
    # Generate synthetic brain image
    brain_image = generator.generate_synthetic_image(seg_data, 'T1').cpu().numpy()
    
    # Generate tumor on the synthetic image
    result = generator.generate_tumor_on_image(
        brain_image, 
        seg_data, 
        modality='T1',
        tumor_size_factor=1.0
    )
    
    # Save results
    output_dir = "output_synthetic_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create identity affine matrix
    affine = np.eye(4)
    
    nib.save(nib.Nifti1Image(brain_image, affine), 
             f"{output_dir}/T1_original.nii.gz")
    nib.save(nib.Nifti1Image(result['diseased_image'], affine), 
             f"{output_dir}/T1_with_tumor.nii.gz")
    nib.save(nib.Nifti1Image(result['tumor_mask'], affine), 
             f"{output_dir}/tumor_mask.nii.gz")
    nib.save(nib.Nifti1Image(result['tumor_prob'], affine), 
             f"{output_dir}/tumor_probability.nii.gz")
    nib.save(nib.Nifti1Image(seg_data, affine), 
             f"{output_dir}/segmentation.nii.gz")
    
    print(f"Results saved to {output_dir}/")
    print(f"Tumor volume: {result['tumor_mask'].sum()} voxels")


def example_multiple_modalities():
    """Example generating tumors for different modalities"""
    print("=== Example with Multiple Modalities ===")
    
    # Create synthetic data
    shape = (96, 96, 96)  # Smaller for faster processing
    seg_data = create_example_segmentation(shape)
    
    # Initialize generator
    generator = SimpleTumorGenerator(device='cpu')
    
    modalities = ['T1', 'T2', 'FLAIR']
    
    for modality in modalities:
        print(f"\nGenerating {modality} tumor...")
        
        # Generate synthetic brain image for this modality
        brain_image = generator.generate_synthetic_image(seg_data, modality).cpu().numpy()
        
        # Generate tumor
        result = generator.generate_tumor_on_image(
            brain_image, 
            seg_data, 
            modality=modality,
            tumor_size_factor=1.2  # Slightly larger tumor
        )
        
        # Save results
        output_dir = f"output_multi_modality/{modality}"
        os.makedirs(output_dir, exist_ok=True)
        
        affine = np.eye(4)
        
        nib.save(nib.Nifti1Image(brain_image, affine), 
                 f"{output_dir}/{modality}_original.nii.gz")
        nib.save(nib.Nifti1Image(result['diseased_image'], affine), 
                 f"{output_dir}/{modality}_with_tumor.nii.gz")
        nib.save(nib.Nifti1Image(result['tumor_mask'], affine), 
                 f"{output_dir}/tumor_mask.nii.gz")
        
        print(f"  Results saved to {output_dir}/")
        print(f"  Tumor volume: {result['tumor_mask'].sum()} voxels")


def main():
    """Run all examples"""
    print("Simple Tumor Generator Examples")
    print("=" * 40)
    
    # Example 1: Synthetic data
    example_with_synthetic_data()
    
    print("\n" + "=" * 40)
    
    # Example 2: Multiple modalities
    example_multiple_modalities()
    
    print("\n" + "=" * 40)
    
    # Example 3: Real data (if available)
    example_with_real_data()
    
    print("\nAll examples completed!")
    print("\nTo visualize the results, you can use:")
    print("- FSLeyes: fsleyes output_*/T1_with_tumor.nii.gz output_*/tumor_mask.nii.gz")
    print("- ITK-SNAP: itksnap output_*/T1_with_tumor.nii.gz -s output_*/tumor_mask.nii.gz")


if __name__ == '__main__':
    main() 