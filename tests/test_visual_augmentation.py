#!/usr/bin/env python3
"""
Domain Randomization Pipeline Visualization for Paper
====================================================

This script creates publication-ready figures showing the domain randomization
augmentation pipeline with difference maps. Perfect for academic papers.

Simply update the HARDCODED_IMAGE_PATH below to point to your test image!
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from brain_age_pred.dataset.augmentation import AugmentationPipeline
from brain_age_pred.dataset.custom_transformations import RandomResolutionD, RandGammaD

# Individual transform imports
from monai.transforms import (
    RandFlipd, RandAffined, RandAdjustContrastd, RandBiasFieldd,
    RandGaussianSmoothd, RandGaussianNoised, RandScaleIntensityd, 
    RandShiftIntensityd, RandHistogramShiftd, RandGibbsNoised, 
    RandCoarseDropoutd, RandSpatialCropd, CenterSpatialCropd,
    ToTensord, LoadImaged, EnsureChannelFirstd, Compose, RandRicianNoised
)

# ====================================================================
# CONFIGURATION - UPDATE THESE PATHS!
# ====================================================================

# Path to your test image (supports .nii.gz, .nii, or .npy files)
HARDCODED_IMAGE_PATH = "/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/CoRR/sub-0003002_ses-1_run-1_T1w.nii.gz"

# Alternative examples (uncomment the one you want to use):
# HARDCODED_IMAGE_PATH = "C:/Projects/thesis_project/Data/brain_age_preprocessed/CamCAN/sub-CC110033_T1w.nii.gz"
# HARDCODED_IMAGE_PATH = "/path/to/your/brain/image.nii.gz"
# HARDCODED_IMAGE_PATH = "/path/to/your/brain/image.npy"

# Path to config file with domain randomization settings
CONFIG_PATH = "brain_age_pred/configs/brainagenext/brainagenext_baseline.yaml"

# ====================================================================


class DomainRandomizationVisualizer:
    """Creates publication-ready visualizations of domain randomization pipeline."""
    
    def __init__(self, config_path: str, image_path: str):
        """
        Initialize the visualizer.
        
        Args:
            config_path: Path to the YAML config file with domain randomization settings
            image_path: Path to the brain image file
        """
        self.config_path = config_path
        self.image_path = image_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dr_config = self.config['domain_randomization']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_key = "image"
        
        # Create output directory
        self.output_dir = Path("brain_age_pred/paper_assets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test image
        self.original_image = self._load_image()
        
        print(f"Using device: {self.device}")
        print(f"Loaded image: {image_path}")
        print(f"Image shape: {self.original_image.shape}")
        print(f"Image range: [{self.original_image.min():.3f}, {self.original_image.max():.3f}]")
        print(f"Output directory: {self.output_dir.absolute()}")
    
    def _load_image(self) -> torch.Tensor:
        """Load and preprocess the brain image."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        print(f"Loading image: {self.image_path}")
        
        if self.image_path.endswith('.npy'):
            # Load numpy array
            img_data = np.load(self.image_path)
            img_tensor = torch.from_numpy(img_data).float()
        elif self.image_path.endswith(('.nii', '.nii.gz')):
            # Load NIfTI file
            nii = nib.load(self.image_path)
            img_data = nii.get_fdata()
            img_tensor = torch.from_numpy(img_data).float()
        else:
            raise ValueError(f"Unsupported file format: {self.image_path}. Use .nii.gz, .nii, or .npy")
        
        # Ensure 4D: (C, H, W, D)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        
        # Normalize to [0, 1]
        img_min, img_max = img_tensor.min(), img_tensor.max()
        if img_max > img_min:
            img_tensor = (img_tensor - img_min) / (img_max - img_min)
        
        return img_tensor.to(self.device)
    
    def _get_center_slice(self, img: torch.Tensor, axis: int = 2) -> np.ndarray:
        """Extract center slice from 3D volume for visualization."""
        if len(img.shape) == 4:  # (C, H, W, D)
            img = img[0]  # Remove channel dimension
        
        center_idx = img.shape[axis] // 2
        if axis == 0:
            slice_img = img[center_idx, :, :]
        elif axis == 1:
            slice_img = img[:, center_idx, :]
        else:  # axis == 2
            slice_img = img[:, :, center_idx]
        
        return slice_img.cpu().numpy()
    
    def _create_domain_randomizer(self, **override_params) -> AugmentationPipeline:
        """Create a domain randomizer with optional parameter overrides."""
        dr_params = {**self.dr_config}
        dr_params.update(override_params)
        dr_params['use_domain_randomization'] = True
        dr_params['device'] = self.device
        dr_params['image_key'] = self.image_key
        
        return AugmentationPipeline(**dr_params)
    
    def _apply_individual_transform(self, transform, title: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Apply a single transform and return result with difference map."""
        sample = {self.image_key: self.original_image.clone()}
        
        # Ensure random transforms are actually applied for visualization
        if hasattr(transform, 'set_random_state'):
            transform.set_random_state(seed=0) # for reproducibility
        
        transformed = transform(sample)[self.image_key]
        
        # Calculate difference map
        original_slice = self._get_center_slice(self.original_image)
        transformed_slice = self._get_center_slice(transformed)
        diff_map = np.abs(transformed_slice - original_slice)
        
        return transformed, diff_map
    
    def create_main_pipeline_figure(self):
        """Create the main figure showing the complete domain randomization pipeline."""
        print("Creating main domain randomization pipeline figure...")
        
        # Define the 8 key transforms from the pipeline for visualization
        transforms_config = [
            {
                'name': 'Original',
                'transform': None,
                'description': 'Original\nImage'
            },
            {
                'name': 'Flip',
                'transform': RandFlipd(
                    keys=[self.image_key], prob=1.0, 
                    spatial_axis=0
                ),
                'description': 'Random\nFlip'
            },
            {
                'name': 'Affine',
                'transform': RandAffined(
                    keys=[self.image_key], prob=1.0,
                    rotate_range=(0.26, 0.26, 0.26),  # ~15 degrees
                    scale_range=(0.15, 0.15, 0.15),   # Corresponds to [0.85, 1.15]
                    mode="bilinear"
                ),
                'description': 'Affine\nTransform'
            },
            {
                'name': 'Contrast',
                'transform': RandAdjustContrastd(
                    keys=[self.image_key], prob=1.0,
                    gamma=(0.5, 2.0) # From config [0.1, 2.0], narrowed for visual effect
                ),
                'description': 'Contrast\nAdjustment'
            },
            {
                'name': 'Gamma',
                'transform': RandGammaD(
                    keys=[self.image_key], prob=1.0,
                    log_gamma_std=0.3
                ),
                'description': 'Gamma\nCorrection'
            },
            {
                'name': 'Bias Field',
                'transform': RandBiasFieldd(
                    keys=[self.image_key], prob=1.0,
                    coeff_range=(0.0, 0.8)
                ),
                'description': 'Bias Field\nArtifact'
            },
            {
                'name': 'Noise',
                'transform': RandGaussianNoised(
                    keys=[self.image_key], prob=1.0,
                    mean=0.0, std=0.08
                ),
                'description': 'Gaussian\nNoise'
            },
            {
                'name': 'Blur',
                'transform': RandGaussianSmoothd(
                    keys=[self.image_key], prob=1.0,
                    sigma_x=(0.5, 2.0), sigma_y=(0.5, 2.0), sigma_z=(0.5, 2.0)
                ),
                'description': 'Gaussian\nBlur'
            }
        ]
        
        # Create figure with 4x4 layout: 2 rows of images, 2 rows of diffs
        n_transforms = len(transforms_config)
        n_cols = 4
        n_rows = ((n_transforms + n_cols - 1) // n_cols) * 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows + 0.5))
        
        # Set overall title
        #fig.suptitle('Domain Randomization Augmentation Pipeline', fontsize=16, fontweight='bold')
        
        original_slice = self._get_center_slice(self.original_image)
        im = None

        for i, config in enumerate(transforms_config):
            row = (i // n_cols) * 2
            col = i % n_cols
            
            transform = config['transform']
            title = config['description']
            
            if transform is None:
                # Original image
                transformed_slice = original_slice
                diff_map = np.zeros_like(original_slice)
            else:
                # Apply transform
                transformed, diff_map = self._apply_individual_transform(transform, title)
                transformed_slice = self._get_center_slice(transformed)
            
            # Plot augmented image (top row of pair)
            axes[row, col].imshow(transformed_slice, cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
            
            # Plot difference map (bottom row of pair)
            if i == 0:
                # For original, show empty difference map
                axes[row + 1, col].imshow(diff_map, cmap='gray', vmin=0, vmax=0.3)
            else:
                # Show actual difference map
                im = axes[row + 1, col].imshow(diff_map, cmap='hot', vmin=0, vmax=0.3)
            axes[row + 1, col].axis('off')
        
        # Add row labels
        if n_rows > 1:
            axes[0, 0].set_ylabel('Augmented', fontsize=12, fontweight='bold', labelpad=20)
            axes[1, 0].set_ylabel('Difference Map', fontsize=12, fontweight='bold', labelpad=20)
        if n_rows > 3:
            axes[2, 0].set_ylabel('Augmented', fontsize=12, fontweight='bold', labelpad=20)
            axes[3, 0].set_ylabel('Difference Map', fontsize=12, fontweight='bold', labelpad=20)

        # Hide any unused subplots
        for i in range(n_transforms, n_cols * (n_rows // 2)):
            row = (i // n_cols) * 2
            col = i % n_cols
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')

        # Add colorbar for difference maps
        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Intensity Difference', rotation=270, labelpad=15)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save figure
        output_path = self.output_dir / 'domain_randomization_pipeline.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Main pipeline figure saved to: {output_path}")
        plt.show()
    
    def create_combined_augmentations_figure(self):
        """Create figure showing different intensities of combined augmentations."""
        print("Creating combined augmentations figure...")
        
        # Define different augmentation intensities
        intensity_configs = [
            {
                'name': 'Original',
                'description': 'Original\nImage',
                'prob_scale': 0.0
            },
            {
                'name': 'Mild',
                'description': 'Mild\nAugmentation',
                'prob_scale': 0.3
            },
            {
                'name': 'Moderate',
                'description': 'Moderate\nAugmentation',
                'prob_scale': 0.6
            },
            {
                'name': 'Strong',
                'description': 'Strong\nAugmentation',
                'prob_scale': 1.0
            },
            {
                'name': 'Combined Sample 1',
                'description': 'Random\nSample 1',
                'prob_scale': 1.0,
                'random': True
            },
            {
                'name': 'Combined Sample 2',
                'description': 'Random\nSample 2',
                'prob_scale': 1.0,
                'random': True
            }
        ]
        
        n_configs = len(intensity_configs)
        n_cols = 3
        n_rows = ((n_configs + n_cols - 1) // n_cols) * 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows + 0.5))
        
        fig.suptitle('Combined Domain Randomization at Different Intensities', fontsize=16, fontweight='bold')
        
        original_slice = self._get_center_slice(self.original_image)
        im = None
        
        for i, config in enumerate(intensity_configs):
            row = (i // n_cols) * 2
            col = i % n_cols

            if config['prob_scale'] == 0.0:
                # Original image
                transformed_slice = original_slice
                diff_map = np.zeros_like(original_slice)
            else:
                # Create domain randomizer with scaled probabilities
                scaled_probs = {
                    k: v * config['prob_scale'] 
                    for k, v in self.dr_config['transform_probs'].items()
                }
                dr = self._create_domain_randomizer(transform_probs=scaled_probs)
                
                # Apply domain randomization
                sample = {self.image_key: self.original_image.clone()}
                result = dr(sample)
                transformed = result[self.image_key]
                
                transformed_slice = self._get_center_slice(transformed)
                diff_map = np.abs(transformed_slice - original_slice)
            
            # Plot augmented image (top row)
            axes[row, col].imshow(transformed_slice, cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(config['description'], fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
            
            # Plot difference map (bottom row)
            if config['prob_scale'] == 0.0:
                axes[row+1, col].imshow(diff_map, cmap='gray', vmin=0, vmax=0.3)
            else:
                im = axes[row+1, col].imshow(diff_map, cmap='hot', vmin=0, vmax=0.3)
            axes[row+1, col].axis('off')

        # Add row labels
        if n_rows > 1:
            axes[0, 0].set_ylabel('Augmented', fontsize=12, fontweight='bold', labelpad=20)
            axes[1, 0].set_ylabel('Difference Map', fontsize=12, fontweight='bold', labelpad=20)
        if n_rows > 3:
            axes[2, 0].set_ylabel('Augmented', fontsize=12, fontweight='bold', labelpad=20)
            axes[3, 0].set_ylabel('Difference Map', fontsize=12, fontweight='bold', labelpad=20)

        # Hide any unused subplots
        for i in range(n_configs, n_cols * (n_rows // 2)):
            row = (i // n_cols) * 2
            col = i % n_cols
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')

        # Add colorbar
        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Intensity Difference', rotation=270, labelpad=15)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save figure
        output_path = self.output_dir / 'combined_augmentations.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Combined augmentations figure saved to: {output_path}")
        plt.show()
    
    def create_noise_artifacts_figure(self):
        """Create figure specifically showing different types of noise and artifacts."""
        print("Creating noise and artifacts figure...")
        
        noise_configs = [
            {
                'name': 'Original',
                'transform': None,
                'description': 'Original\nImage'
            },
            {
                'name': 'Bias Field',
                'transform': RandBiasFieldd(
                    keys=[self.image_key], prob=1.0,
                    coeff_range=(0.0, 0.6)
                ),
                'description': 'Bias Field\nArtifact'
            },
            {
                'name': 'Gaussian Noise',
                'transform': RandGaussianNoised(
                    keys=[self.image_key], prob=1.0,
                    mean=0.0, std=0.08
                ),
                'description': 'Gaussian\nNoise'
            },
            {
                'name': 'Rician Noise',
                'transform': RandRicianNoised(
                    keys=[self.image_key], prob=1.0,
                    std=0.06
                ),
                'description': 'Rician\nNoise'
            },
            {
                'name': 'Gibbs Artifacts',
                'transform': RandGibbsNoised(
                    keys=[self.image_key], prob=1.0,
                    alpha=(0.2, 0.8)
                ),
                'description': 'Gibbs\nArtifacts'
            },
            {
                'name': 'Coarse Dropout',
                'transform': RandCoarseDropoutd(
                    keys=[self.image_key], prob=1.0,
                    holes=6, spatial_size=(25, 25, 25),
                    fill_value=0.0
                ),
                'description': 'Signal\nDropout'
            }
        ]
        
        n_configs = len(noise_configs)
        n_cols = 3
        n_rows = ((n_configs + n_cols - 1) // n_cols) * 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows + 0.5))
        
        fig.suptitle('Noise and Artifact Augmentations', fontsize=16, fontweight='bold')
        
        original_slice = self._get_center_slice(self.original_image)
        im = None
        
        for i, config in enumerate(noise_configs):
            row = (i // n_cols) * 2
            col = i % n_cols

            transform = config['transform']
            title = config['description']
            
            if transform is None:
                transformed_slice = original_slice
                diff_map = np.zeros_like(original_slice)
            else:
                transformed, diff_map = self._apply_individual_transform(transform, title)
                transformed_slice = self._get_center_slice(transformed)
            
            # Plot augmented image (top row)
            axes[row, col].imshow(transformed_slice, cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
            
            # Plot difference map (bottom row)
            if i == 0:
                axes[row + 1, col].imshow(diff_map, cmap='gray', vmin=0, vmax=0.3)
            else:
                im = axes[row + 1, col].imshow(diff_map, cmap='hot', vmin=0, vmax=0.3)
            axes[row + 1, col].axis('off')
        
        # Add row labels
        if n_rows > 1:
            axes[0, 0].set_ylabel('Augmented', fontsize=12, fontweight='bold', labelpad=20)
            axes[1, 0].set_ylabel('Difference Map', fontsize=12, fontweight='bold', labelpad=20)
        if n_rows > 3:
            axes[2, 0].set_ylabel('Augmented', fontsize=12, fontweight='bold', labelpad=20)
            axes[3, 0].set_ylabel('Difference Map', fontsize=12, fontweight='bold', labelpad=20)

        # Hide any unused subplots
        for i in range(n_configs, n_cols * (n_rows // 2)):
            row = (i // n_cols) * 2
            col = i % n_cols
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')

        # Add colorbar
        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Intensity Difference', rotation=270, labelpad=15)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save figure
        output_path = self.output_dir / 'noise_artifacts.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Noise and artifacts figure saved to: {output_path}")
        plt.show()
    
    def create_all_figures(self):
        """Create all visualization figures for the paper."""
        print("Creating all domain randomization figures for paper...")
        print("=" * 60)
        
        self.create_main_pipeline_figure()
        print()
        self.create_combined_augmentations_figure()
        
        print("=" * 60)
        print("All figures created successfully!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("- domain_randomization_pipeline.png")
        print("- combined_augmentations.png")
        print("\nThese figures are ready for inclusion in your paper!")


def main():
    """Main function to create domain randomization visualization figures."""
    
    # Check if files exist
    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found: {CONFIG_PATH}")
        print("Please update CONFIG_PATH at the top of the script.")
        return
    
    if not os.path.exists(HARDCODED_IMAGE_PATH):
        print(f"Image file not found: {HARDCODED_IMAGE_PATH}")
        print("Please update HARDCODED_IMAGE_PATH at the top of the script.")
        print("\nSupported formats: .nii.gz, .nii, .npy")
        return
    
    try:
        # Create visualizer
        visualizer = DomainRandomizationVisualizer(
            config_path=CONFIG_PATH,
            image_path=HARDCODED_IMAGE_PATH
        )
        
        # Create all figures
        visualizer.create_all_figures()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()