"""
Tumor Simulation Module for Domain Randomization
================================================

GPU-aware tumor simulation using Perlin noise that integrates with the 
domain randomization pipeline. Can be used as a transform in the data loader.

Usage:
    tumor_sim = TumorSimulator(device=torch.device("cuda"), **cfg["tumor_simulation"])
    train_ds = BADataset(..., transform=Compose([domain_randomizer, tumor_sim]), mode="train")
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import sys
import os
from pathlib import Path

import numpy as np
import torch
import random
import nibabel as nib
from monai.transforms.transform import MapTransform
import UNA.utils.misc as utils
from UNA.FluidAnomaly.perlin3d import generate_shape_3d, generate_velocity_3d
from UNA.FluidAnomaly.DiffEqs.pde import AdvDiffPDE
from UNA.FluidAnomaly.DiffEqs.odeint import odeint


class AgeBasedSegmentationLoader:
    """
    Loads and manages age-specific brain segmentations for tumor placement.
    """
    
    def __init__(
        self,
        segmentation_paths: Dict[str, str],
        age_ranges: Dict[str, Tuple[float, float]],
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            segmentation_paths: Dict mapping age group names to segmentation file paths
                e.g., {"young": "path/to/young_seg.nii.gz", "middle": "...", "old": "..."}
            age_ranges: Dict mapping age group names to (min_age, max_age) tuples
                e.g., {"young": (18, 40), "middle": (40, 60), "old": (60, 85)}
            device: Device to load segmentations on
        """
        self.age_ranges = age_ranges
        self.device = device
        self.segmentations = {}
        
        # Load all segmentations
        for age_group, seg_path in segmentation_paths.items():
            if not Path(seg_path).exists():
                raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
            
            print(f"Loading {age_group} segmentation from {seg_path}")
            seg_img = nib.load(seg_path)
            seg_data = torch.from_numpy(seg_img.get_fdata()).long().to(device)
            self.segmentations[age_group] = seg_data
            print(f"Loaded {age_group} segmentation with shape {seg_data.shape}")
    
    def get_segmentation_for_age(self, age: float) -> torch.Tensor:
        """
        Get the appropriate segmentation for a given age.
        
        Args:
            age: Age in years
            
        Returns:
            Segmentation tensor for the appropriate age group
        """
        for age_group, (min_age, max_age) in self.age_ranges.items():
            if min_age <= age < max_age:
                return self.segmentations[age_group]
        
        # If age doesn't fit any range, use the closest one
        age_diffs = {}
        for age_group, (min_age, max_age) in self.age_ranges.items():
            mid_age = (min_age + max_age) / 2
            age_diffs[age_group] = abs(age - mid_age)
        
        closest_group = min(age_diffs, key=age_diffs.get)
        print(f"Warning: Age {age} doesn't fit any range, using {closest_group} segmentation")
        return self.segmentations[closest_group]


class TumorSimulator(MapTransform):
    """
    MONAI-compatible tumor simulation transform using Perlin noise.
    
    This transform generates synthetic tumors on brain images and can be used
    as part of the domain randomization pipeline.
    """
    
    def __init__(
        self,
        keys: Union[str, list] = "image",
        *,
        device: torch.device = torch.device("cpu"),
        prob: float = 0.3,
        # Age-based segmentation parameters
        use_age_based_segmentation: bool = False,
        segmentation_paths: Optional[Dict[str, str]] = None,
        age_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        # Tumor generation parameters
        perlin_res: Tuple[int, int, int] = (2, 2, 2),
        mask_percentile_min: float = 90.0,
        mask_percentile_max: float = 99.6,
        tumor_size_factor_range: Tuple[float, float] = (0.5, 2.0),
        pathol_thres: float = 0.2,
        min_tumor_size: int = 100,
        # Fluid dynamics parameters
        use_fluid_dynamics: bool = True,
        V_multiplier: float = 500.0,
        dt: float = 0.1,
        min_nt: int = 10,
        max_nt: int = 20,
        integ_method: str = 'dopri5',
        bc: str = 'neumann',
        # Intensity parameters
        modality: str = 'T1',  # Default modality
        intensity_variation: float = 0.3,
        # Brain mask parameters (if segmentation not available)
        brain_threshold: float = 0.1,  # Threshold for brain tissue detection
        **unused,
    ):
        super().__init__(keys)
        self.device = device
        self.prob = prob
        
        # Store tumor generation parameters
        self.shape_gen_args = {
            'perlin_res': list(perlin_res),
            'mask_percentile_min': mask_percentile_min,
            'mask_percentile_max': mask_percentile_max,
            'pathol_thres': pathol_thres,
            'min_tumor_size': min_tumor_size,
            'integ_method': integ_method,
            'bc': bc,
            'V_multiplier': V_multiplier,
            'dt': dt,
            'min_nt': min_nt,
            'max_nt': max_nt,
        }
        
        self.tumor_size_factor_range = tumor_size_factor_range
        self.use_fluid_dynamics = use_fluid_dynamics
        self.modality = modality
        self.intensity_variation = intensity_variation
        self.brain_threshold = brain_threshold
        
        # Initialize age-based segmentation loader if enabled
        self.use_age_based_segmentation = use_age_based_segmentation
        if self.use_age_based_segmentation:
            if not segmentation_paths or not age_ranges:
                raise ValueError("segmentation_paths and age_ranges must be provided when use_age_based_segmentation=True")
            
            self.seg_loader = AgeBasedSegmentationLoader(
                segmentation_paths=segmentation_paths,
                age_ranges=age_ranges,
                device=device
            )
        else:
            self.seg_loader = None
        
        # Initialize PDE for fluid dynamics if enabled
        if self.use_fluid_dynamics:
            self.t = torch.from_numpy(
                np.arange(self.shape_gen_args['max_nt']) * self.shape_gen_args['dt']
            ).to(self.device)
            
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
        else:
            self.t = None
            self.adv_pde = None
    
    def _generate_tumor_shape(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def _augment_tumor_with_fluid_dynamics(self, tumor_prob: torch.Tensor) -> torch.Tensor:
        """Apply fluid dynamics to make tumor shape more realistic"""
        if not self.use_fluid_dynamics or self.adv_pde is None:
            return tumor_prob
            
        tumor_prob = torch.squeeze(tumor_prob)
        
        # Generate random number of time steps
        nt = np.random.randint(
            self.shape_gen_args['min_nt'], 
            self.shape_gen_args['max_nt'] + 1
        )
        
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
            # If PDE fails, return original
            pass
        
        return tumor_prob
    
    def _get_brain_mask(self, image: torch.Tensor) -> torch.Tensor:
        """Generate brain mask from image intensity"""
        # Simple brain mask based on intensity threshold
        brain_mask = (image > self.brain_threshold).float()
        
        # Remove small connected components (optional)
        # This is a simple approximation - in practice you might want more sophisticated brain extraction
        return brain_mask
    
    def _get_brain_mask_from_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """Generate brain mask from segmentation (GM + WM)"""
        # Brain tissue = Gray Matter (1) + White Matter (2)
        brain_mask = ((segmentation == 1) | (segmentation == 2)).float()
        return brain_mask
    
    def _get_contrast_values(self, modality: str) -> torch.Tensor:
        """Get contrast values for different tissue types based on modality"""
        if modality.upper() == 'T1':
            # T1-weighted: tumors typically hypointense
            base_intensity = 0.4 + 0.3 * torch.rand(1, device=self.device)
        elif modality.upper() == 'T2':
            # T2-weighted: tumors typically hyperintense
            base_intensity = 1.3 + 0.4 * torch.rand(1, device=self.device)
        elif modality.upper() == 'FLAIR':
            # FLAIR: tumors typically hyperintense
            base_intensity = 1.4 + 0.5 * torch.rand(1, device=self.device)
        else:
            # Default to T1-like
            base_intensity = 0.4 + 0.3 * torch.rand(1, device=self.device)
        
        return base_intensity
    
    def _encode_pathology(self, image: torch.Tensor, tumor_prob: torch.Tensor, modality: str) -> torch.Tensor:
        """Encode tumor pathology into the image"""
        # Calculate reference intensity (mean of non-zero regions)
        brain_mask = (image > 0)
        if brain_mask.sum() > 0:
            ref_intensity = (image * brain_mask).sum() / brain_mask.sum()
        else:
            ref_intensity = torch.tensor(1.0, device=self.device)
        
        # Get pathology intensity based on modality
        intensity_multiplier = self._get_contrast_values(modality)
        
        # Add some variation
        intensity_variation = 1.0 + self.intensity_variation * (torch.rand_like(tumor_prob) - 0.5)
        pathol_intensity = ref_intensity * intensity_multiplier * intensity_variation
        
        # Apply pathology
        if modality.upper() in ['T2', 'FLAIR']:
            # Hyperintense lesion (additive)
            diseased_image = image + tumor_prob * pathol_intensity
        else:  # T1
            # Hypointense lesion (multiplicative + additive)
            diseased_image = image * (1 - tumor_prob * 0.6) + tumor_prob * pathol_intensity
        
        # Ensure non-negative values
        diseased_image = torch.clamp(diseased_image, min=0)
        
        return diseased_image
    
    def _generate_tumor_on_image(self, image: torch.Tensor, modality: str, age: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Generate a tumor on the input image
        
        Args:
            image: torch tensor of the brain image (C, H, W, D)
            modality: string, one of 'T1', 'T2', 'FLAIR'
            age: age of the subject (used for age-based segmentation)
        
        Returns:
            dict with 'diseased_image', 'tumor_mask', 'tumor_prob'
        """
        # Remove channel dimension for processing
        if image.dim() == 4 and image.shape[0] == 1:
            image_3d = image.squeeze(0)
            add_channel_dim = True
        else:
            image_3d = image
            add_channel_dim = False
        
        # Generate tumor shape
        tumor_prob, _ = self._generate_tumor_shape(image_3d.shape)
        
        # Apply fluid dynamics augmentation
        tumor_prob = self._augment_tumor_with_fluid_dynamics(tumor_prob)
        
        # Scale tumor size randomly
        tumor_size_factor = random.uniform(*self.tumor_size_factor_range)
        tumor_prob = tumor_prob * tumor_size_factor
        tumor_prob = torch.clamp(tumor_prob, 0, 1)
        
        # Get brain mask (either from age-based segmentation or intensity threshold)
        if self.use_age_based_segmentation and self.seg_loader is not None and age is not None:
            # Use age-based segmentation
            segmentation = self.seg_loader.get_segmentation_for_age(age)
            brain_mask = self._get_brain_mask_from_segmentation(segmentation)
        else:
            # Fall back to intensity-based brain mask
            brain_mask = self._get_brain_mask(image_3d)
        
        # Restrict tumor to brain tissue
        tumor_prob = tumor_prob * brain_mask
        
        # Check if tumor is large enough
        if tumor_prob.sum() < self.shape_gen_args['min_tumor_size']:
            # If tumor too small, try again with larger size factor
            tumor_size_factor = random.uniform(1.5, 3.0)
            tumor_prob = tumor_prob * tumor_size_factor
            tumor_prob = torch.clamp(tumor_prob, 0, 1)
            tumor_prob = tumor_prob * brain_mask
        
        # Create final tumor mask
        tumor_mask = (tumor_prob > self.shape_gen_args['pathol_thres']).float()
        
        # Apply pathology to image
        diseased_image = self._encode_pathology(image_3d, tumor_prob, modality)
        
        # Add channel dimension back if needed
        if add_channel_dim:
            diseased_image = diseased_image.unsqueeze(0)
            tumor_mask = tumor_mask.unsqueeze(0)
            tumor_prob = tumor_prob.unsqueeze(0)
        
        return {
            'diseased_image': diseased_image,
            'tumor_mask': tumor_mask,
            'tumor_prob': tumor_prob
        }
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply tumor simulation to the data sample
        
        Args:
            data: Dictionary containing at least the image key and optionally age
            
        Returns:
            Modified data dictionary with tumor applied (if probability allows)
        """
        d = dict(data)
        
        # Check if we should apply tumor simulation
        if random.random() >= self.prob:
            return d
        
        # Get the image
        key = self.keys[0] if isinstance(self.keys, list) else self.keys
        image = d[key]
        
        # Ensure image is on the correct device
        if image.device != self.device:
            image = image.to(self.device)
        
        # Get age if available
        age = None
        if 'age' in d:
            age_tensor = d['age']
            if isinstance(age_tensor, torch.Tensor):
                age = age_tensor.item()
            else:
                age = float(age_tensor)
        
        # Determine modality (use from data if available, otherwise use default)
        modality = d.get('modality', self.modality)
        
        try:
            # Generate tumor
            result = self._generate_tumor_on_image(image, modality, age)
            
            # Update the image in the data dictionary
            d[key] = result['diseased_image']
            
            # Optionally add tumor mask and probability to the data
            # (useful for debugging or additional processing)
            d['tumor_mask'] = result['tumor_mask']
            d['tumor_prob'] = result['tumor_prob']
            d['has_tumor'] = torch.tensor(True, dtype=torch.bool, device=self.device)
            
        except Exception as e:
            # If tumor generation fails, return original data
            print(f"Warning: Tumor generation failed: {e}")
            d['has_tumor'] = torch.tensor(False, dtype=torch.bool, device=self.device)
        
        return d


# Keep the old classes for backward compatibility
class TumorSimulatorWithSegmentation(TumorSimulator):
    """
    Enhanced tumor simulator that uses brain segmentation for more accurate tumor placement.
    
    Expects segmentation data in the sample with key 'segmentation' or 'seg'.
    Segmentation should have labels: 0=background, 1=GM, 2=WM, 3=CSF
    """
    
    def __init__(self, *args, segmentation_key: str = "segmentation", **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentation_key = segmentation_key
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply tumor simulation using segmentation information
        """
        d = dict(data)
        
        # Check if we should apply tumor simulation
        if random.random() >= self.prob:
            return d
        
        # Get the image and segmentation
        key = self.keys[0] if isinstance(self.keys, list) else self.keys
        image = d[key]
        
        # Check if segmentation is available
        seg_key = None
        for possible_key in [self.segmentation_key, 'seg', 'segmentation']:
            if possible_key in d:
                seg_key = possible_key
                break
        
        if seg_key is None:
            # Fall back to parent class behavior
            return super().__call__(data)
        
        segmentation = d[seg_key]
        
        # Ensure tensors are on the correct device
        if image.device != self.device:
            image = image.to(self.device)
        if segmentation.device != self.device:
            segmentation = segmentation.to(self.device)
        
        # Determine modality
        modality = d.get('modality', self.modality)
        
        try:
            # Remove channel dimension for processing
            if image.dim() == 4 and image.shape[0] == 1:
                image_3d = image.squeeze(0)
                add_channel_dim = True
            else:
                image_3d = image
                add_channel_dim = False
            
            if segmentation.dim() == 4 and segmentation.shape[0] == 1:
                seg_3d = segmentation.squeeze(0)
            else:
                seg_3d = segmentation
            
            # Generate tumor shape
            tumor_prob, _ = self._generate_tumor_shape(image_3d.shape)
            
            # Apply fluid dynamics augmentation
            tumor_prob = self._augment_tumor_with_fluid_dynamics(tumor_prob)
            
            # Scale tumor size randomly
            tumor_size_factor = random.uniform(*self.tumor_size_factor_range)
            tumor_prob = tumor_prob * tumor_size_factor
            tumor_prob = torch.clamp(tumor_prob, 0, 1)
            
            # Restrict tumor to brain tissue using segmentation
            brain_mask = self._get_brain_mask_from_segmentation(seg_3d)
            tumor_prob = tumor_prob * brain_mask
            
            # Check if tumor is large enough
            if tumor_prob.sum() < self.shape_gen_args['min_tumor_size']:
                tumor_size_factor = random.uniform(1.5, 3.0)
                tumor_prob = tumor_prob * tumor_size_factor
                tumor_prob = torch.clamp(tumor_prob, 0, 1)
                tumor_prob = tumor_prob * brain_mask
            
            # Create final tumor mask
            tumor_mask = (tumor_prob > self.shape_gen_args['pathol_thres']).float()
            
            # Apply pathology to image
            diseased_image = self._encode_pathology(image_3d, tumor_prob, modality)
            
            # Add channel dimension back if needed
            if add_channel_dim:
                diseased_image = diseased_image.unsqueeze(0)
                tumor_mask = tumor_mask.unsqueeze(0)
                tumor_prob = tumor_prob.unsqueeze(0)
            
            # Update the data dictionary
            d[key] = diseased_image
            d['tumor_mask'] = tumor_mask
            d['tumor_prob'] = tumor_prob
            d['has_tumor'] = torch.tensor(True, dtype=torch.bool, device=self.device)
            
        except Exception as e:
            print(f"Warning: Tumor generation with segmentation failed: {e}")
            d['has_tumor'] = torch.tensor(False, dtype=torch.bool, device=self.device)
        
        return d