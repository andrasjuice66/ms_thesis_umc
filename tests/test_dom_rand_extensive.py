#!/usr/bin/env python3
"""
Domain Randomization Transformation Testing Script

This script allows you to test and visually inspect all domain randomization 
transformations with different parameter values (mean, mild, extreme).
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from brain_age_pred.dataset.domain_randomization import DomainRandomizer
from brain_age_pred.dataset.custom_transformations import RandomResolutionD, RandGammaD

# Individual transform imports
from monai.transforms import (
    RandFlipd, RandAffined, RandAdjustContrastd, RandBiasFieldd,
    RandGaussianSmoothd, RandGaussianNoised, RandRicianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandHistogramShiftd,
    RandGibbsNoised, RandCoarseDropoutd, RandSpatialCropd, CenterSpatialCropd,
    ToTensord, LoadImaged, EnsureChannelFirstd, Compose
)
import torchio as tio

class TransformationTester:
    """Test individual transformations with different parameter values."""
  
    def __init__(self, config_path: str, image_path: str):
        """
        Initialize the tester.
      
        Args:
            config_path: Path to the YAML config file
            image_path: Path to the NIfTI image file
        """
        # Hardcoded image path - change this to your test image
        self.image_path = "C:/Projects/thesis_project/Data/brain_age_preprocessed/CamCAN/sub-CC110033_T1w.nii.gz"
      
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
      
        self.dr_config = self.config['domain_randomization']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_key = "image"
      
        # Create output directory for test images
        self.output_dir = Path("brain_age_pred/tests/extensive_tests")
        self.output_dir.mkdir(parents=True, exist_ok=True)
      
        # Load and prepare image
        self.original_image = self._load_image()
      
        print(f"Using device: {self.device}")
        print(f"Image shape: {self.original_image.shape}")
        print(f"Image range: [{self.original_image.min():.3f}, {self.original_image.max():.3f}]")
        print(f"Output directory: {self.output_dir.absolute()}")
  
    def _load_image(self) -> torch.Tensor:
        """Load and preprocess the NIfTI image."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
      
        # Load with nibabel
        nii = nib.load(self.image_path)
        img_data = nii.get_fdata()
      
        # Convert to tensor and add channel dimension
        img_tensor = torch.from_numpy(img_data).float()
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
      
        # Normalize to [0, 1] range
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
  
    def test_geometric_transforms(self):
        """Test geometric transformations: flip, affine, and cropping."""
        print("\n=== Testing Geometric Transformations ===")
      
        # 1. Flip Transform
        print("\n1. Testing Flip Transform")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
      
        # Original
        orig_slice = self._get_center_slice(self.original_image)
        axes[0].imshow(orig_slice, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
      
        # Flip X-axis
        flip_transform = RandFlipd(keys=[self.image_key], prob=1.0, spatial_axis=0)
        sample = {self.image_key: self.original_image.clone()}
        flipped = flip_transform(sample)[self.image_key]
        flip_slice = self._get_center_slice(flipped)
        axes[1].imshow(flip_slice, cmap='gray')
        axes[1].set_title('Flipped (X-axis)')
        axes[1].axis('off')
      
        # Flip Y-axis
        flip_transform_y = RandFlipd(keys=[self.image_key], prob=1.0, spatial_axis=1)
        sample = {self.image_key: self.original_image.clone()}
        flipped_y = flip_transform_y(sample)[self.image_key]
        flip_slice_y = self._get_center_slice(flipped_y)
        axes[2].imshow(flip_slice_y, cmap='gray')
        axes[2].set_title('Flipped (Y-axis)')
        axes[2].axis('off')
      
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_flip_transforms.png', dpi=150, bbox_inches='tight')
        plt.show()
      
        # 2. Affine Transforms
        print("\n2. Testing Affine Transforms")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
      
        # Original
        axes[0, 0].imshow(orig_slice, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
      
        # Mild rotation
        deg2rad = np.pi / 180
        mild_rotation = self.dr_config['rotation_range'] * 0.3  # 30% of max
        affine_mild = RandAffined(
            keys=[self.image_key], prob=1.0,
            rotate_range=(deg2rad * mild_rotation,) * 3,
            mode="bilinear"
        )
        sample = {self.image_key: self.original_image.clone()}
        rotated_mild = affine_mild(sample)[self.image_key]
        axes[0, 1].imshow(self._get_center_slice(rotated_mild), cmap='gray')
        axes[0, 1].set_title(f'Mild Rotation ({mild_rotation:.1f}°)')
        axes[0, 1].axis('off')
      
        # Strong rotation
        strong_rotation = self.dr_config['rotation_range']
        affine_strong = RandAffined(
            keys=[self.image_key], prob=1.0,
            rotate_range=(deg2rad * strong_rotation,) * 3,
            mode="bilinear"
        )
        sample = {self.image_key: self.original_image.clone()}
        rotated_strong = affine_strong(sample)[self.image_key]
        axes[0, 2].imshow(self._get_center_slice(rotated_strong), cmap='gray')
        axes[0, 2].set_title(f'Strong Rotation ({strong_rotation:.1f}°)')
        axes[0, 2].axis('off')
      
        # Mild scaling
        scale_range = self.dr_config['scaling_range']
        mild_scale = (scale_range[1] - 1) * 0.3  # 30% of max scaling
        affine_scale_mild = RandAffined(
            keys=[self.image_key], prob=1.0,
            scale_range=(mild_scale,) * 3,
            mode="bilinear"
        )
        sample = {self.image_key: self.original_image.clone()}
        scaled_mild = affine_scale_mild(sample)[self.image_key]
        axes[1, 0].imshow(self._get_center_slice(scaled_mild), cmap='gray')
        axes[1, 0].set_title(f'Mild Scaling ({1+mild_scale:.2f}x)')
        axes[1, 0].axis('off')
      
        # Strong scaling
        strong_scale = scale_range[1] - 1
        affine_scale_strong = RandAffined(
            keys=[self.image_key], prob=1.0,
            scale_range=(strong_scale,) * 3,
            mode="bilinear"
        )
        sample = {self.image_key: self.original_image.clone()}
        scaled_strong = affine_scale_strong(sample)[self.image_key]
        axes[1, 1].imshow(self._get_center_slice(scaled_strong), cmap='gray')
        axes[1, 1].set_title(f'Strong Scaling ({1+strong_scale:.2f}x)')
        axes[1, 1].axis('off')
      
        # Shearing
        shear_bounds = self.dr_config['shearing_bounds']
        affine_shear = RandAffined(
            keys=[self.image_key], prob=1.0,
            shear_range=(shear_bounds,) * 3,
            mode="bilinear"
        )
        sample = {self.image_key: self.original_image.clone()}
        sheared = affine_shear(sample)[self.image_key]
        axes[1, 2].imshow(self._get_center_slice(sheared), cmap='gray')
        axes[1, 2].set_title(f'Shearing ({shear_bounds:.3f})')
        axes[1, 2].axis('off')
      
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_affine_transforms.png', dpi=150, bbox_inches='tight')
        plt.show()
      
        # 3. Cropping Transforms - NEW SECTION
        print("\n3. Testing Cropping Transforms")
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
      
        # Original
        axes[0, 0].imshow(orig_slice, cmap='gray')
        axes[0, 0].set_title(f'Original Shape: {self.original_image.shape[1:]}')
        axes[0, 0].axis('off')
      
        # Type 1: Random Spatial Crop to target size (160, 192, 160)
        target_size = (160, 192, 160)
        spatial_crop = RandSpatialCropd(
            keys=[self.image_key], 
            roi_size=target_size, 
            random_center=True,
        )
        sample = {self.image_key: self.original_image.clone()}
        cropped_spatial = spatial_crop(sample)[self.image_key]
        axes[0, 1].imshow(self._get_center_slice(cropped_spatial), cmap='gray')
        axes[0, 1].set_title(f'Random Spatial Crop\nShape: {cropped_spatial.shape[1:]}')
        axes[0, 1].axis('off')
      
        center_crop = CenterSpatialCropd(
            keys=[self.image_key], 
            roi_size=target_size
        )
        sample = {self.image_key: self.original_image.clone()}
        cropped_center = center_crop(sample)[self.image_key]
        axes[1, 0].imshow(self._get_center_slice(cropped_center), cmap='gray')
        axes[1, 0].set_title(f'Center Crop\nShape: {cropped_center.shape[1:]}')
        axes[1, 0].axis('off')
      
        # Type 3: Random Crop with different size then resize to target
        from monai.transforms import Resized
        # First crop to a smaller random size
        smaller_size = (140, 170, 140)
        random_crop_small = RandSpatialCropd(
            keys=[self.image_key], 
            roi_size=smaller_size, 
            random_center=True,
        )
        # Then resize to target
        resize_transform = Resized(
            keys=[self.image_key], 
            spatial_size=target_size,
            mode="trilinear"
        )
      
        sample = {self.image_key: self.original_image.clone()}
        cropped_small = random_crop_small(sample)[self.image_key]
        sample_resized = {self.image_key: cropped_small}
        cropped_resized = resize_transform(sample_resized)[self.image_key]
        axes[1, 1].imshow(self._get_center_slice(cropped_resized), cmap='gray')
        axes[1, 1].set_title(f'Crop + Resize\nShape: {cropped_resized.shape[1:]}')
        axes[1, 1].axis('off')
      
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_cropping_transforms.png', dpi=150, bbox_inches='tight')
        plt.show()
  
        # Print shape information
        print(f"\nShape Information:")
        print(f"Original image shape: {self.original_image.shape}")
        print(f"Target shape: {target_size}")
        print(f"Random spatial crop result: {cropped_spatial.shape}")
        print(f"Center crop result: {cropped_center.shape}")
        print(f"Crop + resize result: {cropped_resized.shape}")
  
    def test_intensity_transforms(self):
        """Test intensity transformations."""
        print("\n=== Testing Intensity Transformations ===")
      
        # 1. Contrast and Gamma
        print("\n1. Testing Contrast and Gamma")
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
      
        orig_slice = self._get_center_slice(self.original_image)
      
        # Original
        axes[0, 0].imshow(orig_slice, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
      
        # Contrast variations
        contrast_range = self.dr_config['contrast_range']
      
        # Low contrast
        contrast_low = RandAdjustContrastd(
            keys=[self.image_key], prob=1.0, gamma=(contrast_range[0], contrast_range[0])
        )
        sample = {self.image_key: self.original_image.clone()}
        low_contrast = contrast_low(sample)[self.image_key]
        axes[0, 1].imshow(self._get_center_slice(low_contrast), cmap='gray')
        axes[0, 1].set_title(f'Low Contrast ({contrast_range[0]:.1f})')
        axes[0, 1].axis('off')
      
        # High contrast
        contrast_high = RandAdjustContrastd(
            keys=[self.image_key], prob=1.0, gamma=(contrast_range[1], contrast_range[1])
        )
        sample = {self.image_key: self.original_image.clone()}
        high_contrast = contrast_high(sample)[self.image_key]
        axes[0, 2].imshow(self._get_center_slice(high_contrast), cmap='gray')
        axes[0, 2].set_title(f'High Contrast ({contrast_range[1]:.1f})')
        axes[0, 2].axis('off')
      
        # Gamma correction
        log_gamma_std = self.dr_config['log_gamma_std']
        gamma_transform = RandGammaD(
            keys=[self.image_key], log_gamma_std=log_gamma_std, prob=1.0
        )
        sample = {self.image_key: self.original_image.clone()}
        gamma_corrected = gamma_transform(sample)[self.image_key]
        axes[0, 3].imshow(self._get_center_slice(gamma_corrected), cmap='gray')
        axes[0, 3].set_title(f'Gamma (std={log_gamma_std:.2f})')
        axes[0, 3].axis('off')
      
        # Intensity scaling and shifting
        scale_transform = RandScaleIntensityd(
            keys=[self.image_key], prob=1.0, factors=contrast_range
        )
        sample = {self.image_key: self.original_image.clone()}
        scaled_intensity = scale_transform(sample)[self.image_key]
        axes[1, 0].imshow(self._get_center_slice(scaled_intensity), cmap='gray')
        axes[1, 0].set_title('Intensity Scaling')
        axes[1, 0].axis('off')
      
        # Intensity shifting
        shift_offsets = self.dr_config.get('shift_intensity_offsets', [-0.1, 0.1])
        shift_transform = RandShiftIntensityd(
            keys=[self.image_key], prob=1.0, offsets=shift_offsets
        )
        sample = {self.image_key: self.original_image.clone()}
        shifted_intensity = shift_transform(sample)[self.image_key]
        axes[1, 1].imshow(self._get_center_slice(shifted_intensity), cmap='gray')
        axes[1, 1].set_title('Intensity Shifting')
        axes[1, 1].axis('off')
      
        # Histogram shifting
        hist_points = self.dr_config.get('histogram_num_control_points', [5, 10])
        hist_transform = RandHistogramShiftd(
            keys=[self.image_key], prob=1.0, num_control_points=hist_points
        )
        sample = {self.image_key: self.original_image.clone()}
        hist_shifted = hist_transform(sample)[self.image_key]
        axes[1, 2].imshow(self._get_center_slice(hist_shifted), cmap='gray')
        axes[1, 2].set_title('Histogram Shifting')
        axes[1, 2].axis('off')
      
        # Bias field
        bias_range = self.dr_config['bias_field_range']
        bias_transform = RandBiasFieldd(
            keys=[self.image_key], prob=1.0, coeff_range=bias_range
        )
        sample = {self.image_key: self.original_image.clone()}
        bias_corrected = bias_transform(sample)[self.image_key]
        axes[1, 3].imshow(self._get_center_slice(bias_corrected), cmap='gray')
        axes[1, 3].set_title(f'Bias Field ({bias_range[1]:.1f})')
        axes[1, 3].axis('off')
      
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_intensity_transforms.png', dpi=150, bbox_inches='tight')
        plt.show()
  
    def _create_domain_randomizer(self, **override_params):
        """Helper method to create DomainRandomizer with proper parameter handling."""
        # Start with base config, excluding parameters that shouldn't be passed to DomainRandomizer
        config_copy = {k: v for k, v in self.dr_config.items() 
                      if k not in ['use_domain_randomization', 'augmentation_strength', 
                                   'enable_spatial', 'enable_simulation']}
      
        # Apply any overrides
        config_copy.update(override_params)
      
        return DomainRandomizer(device=self.device, **config_copy)

    def test_noise_and_artifacts(self):
        """Test noise and artifact transformations."""
        print("\n=== Testing Noise and Artifacts ===")
      
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
      
        orig_slice = self._get_center_slice(self.original_image)
      
        # Original
        axes[0, 0].imshow(orig_slice, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
      
        # Gaussian noise
        noise_transform = RandGaussianNoised(
            keys=[self.image_key], prob=1.0, mean=0.0, std=0.05
        )
        sample = {self.image_key: self.original_image.clone()}
        noisy = noise_transform(sample)[self.image_key]
        axes[0, 1].imshow(self._get_center_slice(noisy), cmap='gray')
        axes[0, 1].set_title('Gaussian Noise')
        axes[0, 1].axis('off')
      
        # Rician noise
        rician_transform = RandRicianNoised(
            keys=[self.image_key], prob=1.0, std=0.05
        )
        sample = {self.image_key: self.original_image.clone()}
        rician_noisy = rician_transform(sample)[self.image_key]
        axes[0, 2].imshow(self._get_center_slice(rician_noisy), cmap='gray')
        axes[0, 2].set_title('Rician Noise')
        axes[0, 2].axis('off')
      
        # Gibbs artifacts
        gibbs_transform = RandGibbsNoised(
            keys=[self.image_key], prob=1.0, alpha=(0.0, 1.0)
        )
        sample = {self.image_key: self.original_image.clone()}
        gibbs_artifact = gibbs_transform(sample)[self.image_key]
        axes[0, 3].imshow(self._get_center_slice(gibbs_artifact), cmap='gray')
        axes[0, 3].set_title('Gibbs Artifacts')
        axes[0, 3].axis('off')
      
        # Gaussian blur
        blur_transform = RandGaussianSmoothd(
            keys=[self.image_key], prob=1.0,
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)
        )
        sample = {self.image_key: self.original_image.clone()}
        blurred = blur_transform(sample)[self.image_key]
        axes[1, 0].imshow(self._get_center_slice(blurred), cmap='gray')
        axes[1, 0].set_title('Gaussian Blur')
        axes[1, 0].axis('off')
      
        # Resolution degradation
        max_res_iso = self.dr_config['max_res_iso']
        resolution_transform = RandomResolutionD(
            keys=[self.image_key], min_res=1.0, max_res_iso=max_res_iso, prob=1.0
        )
        sample = {self.image_key: self.original_image.clone()}
        low_res = resolution_transform(sample)[self.image_key]
        axes[1, 1].imshow(self._get_center_slice(low_res), cmap='gray')
        axes[1, 1].set_title(f'Low Resolution (max={max_res_iso:.1f})')
        axes[1, 1].axis('off')
      
        # Coarse dropout
        coarse_size = self.dr_config['coarse_dropout_size']
        dropout_transform = RandCoarseDropoutd(
            keys=[self.image_key], prob=1.0, holes=8,
            spatial_size=coarse_size, fill_value=0.0
        )
        sample = {self.image_key: self.original_image.clone()}
        dropout = dropout_transform(sample)[self.image_key]
        axes[1, 2].imshow(self._get_center_slice(dropout), cmap='gray')
        axes[1, 2].set_title(f'Coarse Dropout ({coarse_size})')
        axes[1, 2].axis('off')
      
        # Combined mild transforms
        mild_dr = self._create_domain_randomizer(
            transform_probs={k: 0.5 for k in self.dr_config['transform_probs']}
        )
        sample = {self.image_key: self.original_image.clone()}
        mild_combined = mild_dr(sample)[self.image_key]
        axes[1, 3].imshow(self._get_center_slice(mild_combined), cmap='gray')
        axes[1, 3].set_title('Combined (Mild)')
        axes[1, 3].axis('off')
      
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_noise_artifacts.png', dpi=150, bbox_inches='tight')
        plt.show()
  
    def test_torchio_transforms(self):
        """Test TorchIO-specific transforms if enabled."""
        if not self.dr_config.get('use_torchio', False):
            print("\n=== TorchIO transforms disabled in config ===")
            return
      
        print("\n=== Testing TorchIO Transformations ===")
      
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
      
        orig_slice = self._get_center_slice(self.original_image)
      
        # Original
        axes[0].imshow(orig_slice, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
      
        # Convert to TorchIO format
        tio_img = tio.ScalarImage(tensor=self.original_image.cpu())
        subject = tio.Subject({self.image_key: tio_img})
      
        # Elastic deformation
        elastic_transform = tio.RandomElasticDeformation(
            num_control_points=7, max_displacement=5.0, p=1.0
        )
        elastic_subject = elastic_transform(subject)
        elastic_img = elastic_subject[self.image_key].data.to(self.device)
        axes[1].imshow(self._get_center_slice(elastic_img), cmap='gray')
        axes[1].set_title('Elastic Deformation')
        axes[1].axis('off')
      
        # Spike artifacts
        spike_transform = tio.RandomSpike(
            num_spikes=(1, 6), intensity=(0.1, 0.6), p=1.0
        )
        spike_subject = spike_transform(subject)
        spike_img = spike_subject[self.image_key].data.to(self.device)
        axes[2].imshow(self._get_center_slice(spike_img), cmap='gray')
        axes[2].set_title('Spike Artifacts')
        axes[2].axis('off')
      
        # Ghosting artifacts
        ghost_transform = tio.RandomGhosting(
            num_ghosts=(2, 10), axes=(0, 1, 2), p=1.0
        )
        ghost_subject = ghost_transform(subject)
        ghost_img = ghost_subject[self.image_key].data.to(self.device)
        axes[3].imshow(self._get_center_slice(ghost_img), cmap='gray')
        axes[3].set_title('Ghosting Artifacts')
        axes[3].axis('off')
      
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_torchio_transforms.png', dpi=150, bbox_inches='tight')
        plt.show()
  
    def test_parameter_variations(self):
        """Test the same transform with different parameter intensities."""
        print("\n=== Testing Parameter Variations ===")
      
        # Test contrast with different intensities
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
      
        orig_slice = self._get_center_slice(self.original_image)
      
        # Row 1: Contrast variations
        contrast_values = [0.5, 0.8, 1.2, 1.8]  # weak to extreme
        for i, contrast in enumerate(contrast_values):
            if i == 0:
                axes[0, i].imshow(orig_slice, cmap='gray')
                axes[0, i].set_title('Original')
            else:
                contrast_transform = RandAdjustContrastd(
                    keys=[self.image_key], prob=1.0, gamma=(contrast, contrast)
                )
                sample = {self.image_key: self.original_image.clone()}
                transformed = contrast_transform(sample)[self.image_key]
                axes[0, i].imshow(self._get_center_slice(transformed), cmap='gray')
                axes[0, i].set_title(f'Contrast {contrast:.1f}')
            axes[0, i].axis('off')
      
        # Row 2: Noise variations
        noise_stds = [0.0, 0.02, 0.05, 0.1]  # no noise to extreme
        for i, std in enumerate(noise_stds):
            if i == 0:
                axes[1, i].imshow(orig_slice, cmap='gray')
                axes[1, i].set_title('No Noise')
            else:
                noise_transform = RandGaussianNoised(
                    keys=[self.image_key], prob=1.0, mean=0.0, std=std
                )
                sample = {self.image_key: self.original_image.clone()}
                noisy = noise_transform(sample)[self.image_key]
                axes[1, i].imshow(self._get_center_slice(noisy), cmap='gray')
                axes[1, i].set_title(f'Noise σ={std:.3f}')
            axes[1, i].axis('off')
      
        # Row 3: Blur variations
        blur_sigmas = [(0, 0, 0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
        for i, sigma in enumerate(blur_sigmas):
            if i == 0:
                axes[2, i].imshow(orig_slice, cmap='gray')
                axes[2, i].set_title('No Blur')
            else:
                blur_transform = RandGaussianSmoothd(
                    keys=[self.image_key], prob=1.0,
                    sigma_x=sigma, sigma_y=sigma, sigma_z=sigma
                )
                sample = {self.image_key: self.original_image.clone()}
                blurred = blur_transform(sample)[self.image_key]
                axes[2, i].imshow(self._get_center_slice(blurred), cmap='gray')
                axes[2, i].set_title(f'Blur σ={sigma[0]:.1f}')
            axes[2, i].axis('off')
      
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_parameter_variations.png', dpi=150, bbox_inches='tight')
        plt.show()
  
    def test_full_pipeline(self):
        """Test the complete domain randomization pipeline."""
        print("\n=== Testing Full Pipeline ===")
      
        # Create domain randomizers with different intensities
        configs = {
            'mild': {'transform_probs': {k: v * 0.3 for k, v in self.dr_config['transform_probs'].items()}},
            'medium': {'transform_probs': {k: v * 0.6 for k, v in self.dr_config['transform_probs'].items()}},
            'strong': {},  # Use default config
            'extreme': {'transform_probs': {k: min(1.0, v * 1.5) for k, v in self.dr_config['transform_probs'].items()}}
        }
      
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
      
        orig_slice = self._get_center_slice(self.original_image)
      
        # Original
        axes[0, 0].imshow(orig_slice, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
      
        # Test different intensities
        for i, (name, config_override) in enumerate(configs.items()):
            dr = self._create_domain_randomizer(**config_override)
          
            sample = {self.image_key: self.original_image.clone()}
            transformed = dr(sample)[self.image_key]
          
            row = 0 if i < 3 else 1
            col = (i % 3) + 1 if i < 3 else i - 3
          
            axes[row, col].imshow(self._get_center_slice(transformed), cmap='gray')
            axes[row, col].set_title(f'{name.capitalize()} Augmentation')
            axes[row, col].axis('off')
      
        # Show multiple samples with same config
        for i in range(3):
            dr = self._create_domain_randomizer()
            sample = {self.image_key: self.original_image.clone()}
            transformed = dr(sample)[self.image_key]
          
            axes[1, i + 1].imshow(self._get_center_slice(transformed), cmap='gray')
            axes[1, i + 1].set_title(f'Random Sample {i + 1}')
            axes[1, i + 1].axis('off')
      
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_full_pipeline.png', dpi=150, bbox_inches='tight')
        plt.show()
  
    def run_all_tests(self):
        """Run all transformation tests."""
        print("Starting Domain Randomization Testing...")
        print(f"Config loaded from: {self.config_path}")
        print(f"Test image: {self.image_path}")
      
        try:
            self.test_geometric_transforms()
            self.test_intensity_transforms()
            self.test_noise_and_artifacts()
            self.test_torchio_transforms()
            self.test_parameter_variations()
            self.test_full_pipeline()
          
            print("\n=== All tests completed successfully! ===")
            print(f"Generated images saved to: {self.output_dir.absolute()}")
            print("Generated images:")
            print("- test_flip_transforms.png")
            print("- test_affine_transforms.png")
            print("- test_intensity_transforms.png")
            print("- test_noise_artifacts.png")
            print("- test_torchio_transforms.png")
            print("- test_parameter_variations.png")
            print("- test_full_pipeline.png")
          
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the tests."""
    # Paths - modify these as needed
    config_path = "C:/Projects/thesis_project/brain_age_pred/configs/sfcn/sfcn_dom_rand_tuning.yaml"
  
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Please update the config_path in the script.")
        return
  
    # Create tester and run tests
    tester = TransformationTester(config_path, "")  # image path is hardcoded in class
    tester.run_all_tests()

if __name__ == "__main__":
    main()