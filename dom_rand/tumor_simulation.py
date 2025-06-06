from __future__ import annotations
from typing import Dict, Optional, Tuple, Union, List
import sys
import os
from pathlib import Path
import logging

import numpy as np
import torch
import random
import nibabel as nib
from monai.transforms.transform import MapTransform
import torch.nn.functional as F     # <-- one new import at top of file
import traceback
from brain_age_pred.dom_rand.FluidAnomaly.perlin3d import generate_shape_3d, generate_velocity_3d
from brain_age_pred.dom_rand.FluidAnomaly.DiffEqs.pde import AdvDiffPDE
from brain_age_pred.dom_rand.FluidAnomaly.DiffEqs.odeint import odeint


class AgeBasedSegmentationLoader:
    """
    Loads and manages age-specific brain segmentations for tumor placement.
    Supports both .npy and NIfTI file formats.
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
                e.g., {"young": "path/to/young_seg.nii.npy", "middle": "...", "old": "..."}
            age_ranges: Dict mapping age group names to (min_age, max_age) tuples
                e.g., {"young": (18, 40), "middle": (40, 60), "old": (60, 85)}
            device: Device to load segmentations on
        """
        self.age_ranges = age_ranges
        self.device = device
        self.segmentations = {}
        
        # Validate inputs
        if not segmentation_paths:
            raise ValueError("segmentation_paths cannot be empty")
        if not age_ranges:
            raise ValueError("age_ranges cannot be empty")
        if set(segmentation_paths.keys()) != set(age_ranges.keys()):
            raise ValueError("segmentation_paths and age_ranges must have the same keys")
        
        # Load all segmentations
        for age_group, seg_path in segmentation_paths.items():
            self._load_segmentation(age_group, seg_path)
    
    def _load_segmentation(self, age_group: str, seg_path: str):
        """Load a single segmentation file"""
        seg_path = Path(seg_path)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        
        print(f"Loading {age_group} segmentation from {seg_path}")
        
        try:
            # Handle different file formats
            if seg_path.suffix == '.npy':
                seg_data = np.load(seg_path)
            elif seg_path.suffix in ['.nii', '.gz'] or str(seg_path).endswith('.nii.gz'):
                seg_img = nib.load(seg_path)
                seg_data = seg_img.get_fdata()
            else:
                raise ValueError(f"Unsupported file format: {seg_path.suffix}")
            
            # Convert to tensor
            seg_tensor = torch.from_numpy(seg_data.astype(np.int64)).to(self.device)
            self.segmentations[age_group] = seg_tensor
            
            print(f"Loaded {age_group} segmentation with shape {seg_tensor.shape}, "
                  f"unique values: {torch.unique(seg_tensor).cpu().numpy()}")
                  
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation {seg_path}: {e}")
    
    def get_segmentation_for_age(self, age: float) -> torch.Tensor:
        """
        Get the appropriate segmentation for a given age.
        
        Args:
            age: Age in years
            
        Returns:
            Segmentation tensor for the appropriate age group
        """
        # Find exact match first
        for age_group, (min_age, max_age) in self.age_ranges.items():
            if min_age <= age < max_age:
                return self.segmentations[age_group]
        
        # Handle edge case where age equals max_age of the last group
        for age_group, (min_age, max_age) in self.age_ranges.items():
            if age == max_age:
                return self.segmentations[age_group]
        
        # If age doesn't fit any range, use the closest one
        age_diffs = {}
        for age_group, (min_age, max_age) in self.age_ranges.items():
            mid_age = (min_age + max_age) / 2
            age_diffs[age_group] = abs(age - mid_age)
        
        closest_group = min(age_diffs, key=age_diffs.get)
        print(f"Warning: Age {age} doesn't fit any range, using {closest_group} segmentation")
        return self.segmentations[closest_group]
    
    def get_available_age_groups(self) -> List[str]:
        """Get list of available age groups"""
        return list(self.segmentations.keys())


class TumorIntensityManager:
    """
    Manages tumor intensity values based on MRI modality.
    """
    
    # Intensity multipliers for different modalities relative to normal tissue
    MODALITY_INTENSITIES = {
        'T1': {
            'base_range': (0.3, 0.6),      # Hypointense
            'variation': 0.2,
            'description': 'Hypointense lesions'
        },
        'T2': {
            'base_range': (1.2, 1.8),      # Hyperintense
            'variation': 0.3,
            'description': 'Hyperintense lesions'
        },
        'FLAIR': {
            'base_range': (1.3, 2.0),      # Hyperintense
            'variation': 0.4,
            'description': 'Hyperintense lesions with CSF suppression'
        },
    }
    
    @classmethod
    def get_tumor_intensity(cls, modality: str, reference_intensity: torch.Tensor, 
                           device: torch.device) -> torch.Tensor:
        """
        Get tumor intensity based on modality and reference tissue intensity.
        
        Args:
            modality: MRI modality ('T1', 'T2', 'FLAIR', etc.)
            reference_intensity: Reference intensity from normal brain tissue
            device: Device for tensor operations
            
        Returns:
            Tumor intensity multiplier
        """
        modality_upper = modality.upper()
        
        if modality_upper not in cls.MODALITY_INTENSITIES:
            print(f"Warning: Unknown modality '{modality}', using T1 defaults")
            modality_upper = 'T1'
        
        intensity_config = cls.MODALITY_INTENSITIES[modality_upper]
        
        # Sample base intensity
        base_min, base_max = intensity_config['base_range']
        base_intensity = torch.rand(1, device=device) * (base_max - base_min) + base_min
        
        # Add variation
        variation = intensity_config['variation']
        intensity_variation = 1.0 + variation * (torch.rand(1, device=device) - 0.5)
        
        final_intensity = base_intensity * intensity_variation
        
        return final_intensity


class TumorShapeGenerator:
    """
    Generates tumor shapes using Perlin noise and optional fluid dynamics.
    """
    
    def __init__(
        self,
        device: torch.device,
        perlin_res: List[int] = [8, 8, 8],
        mask_percentile_min: float = 90.0,
        mask_percentile_max: float = 99.6,
        tumor_size_factor_range: Tuple[float, float] = (0.5, 2.0),
        pathol_thres: float = 0.2,
        min_tumor_size: int = 100,
        use_fluid_dynamics: bool = True,
        V_multiplier: float = 500.0,
        dt: float = 0.1,
        min_nt: int = 10,
        max_nt: int = 20,
        integ_method: str = 'dopri5',
        bc: str = 'neumann',
    ):
        self.device = device
        self.perlin_res = perlin_res
        self.mask_percentile_min = mask_percentile_min
        self.mask_percentile_max = mask_percentile_max
        self.tumor_size_factor_range = tumor_size_factor_range
        self.pathol_thres = pathol_thres
        self.min_tumor_size = min_tumor_size
        self.use_fluid_dynamics = use_fluid_dynamics
        
        # Fluid dynamics parameters
        if self.use_fluid_dynamics:
            self.V_multiplier = V_multiplier
            self.dt = dt
            self.min_nt = min_nt
            self.max_nt = max_nt
            self.integ_method = integ_method
            self.bc = bc
            
            # Initialize PDE solver
            self.t = torch.arange(max_nt, dtype=torch.float32, device=device) * dt
            self.adv_pde = AdvDiffPDE(
                data_spacing=[1., 1., 1.], 
                perf_pattern='adv', 
                V_type='vector_div_free', 
                V_dict={},
                BC=bc, 
                dt=dt, 
                device=device
            )
    
    def generate_tumor_shape(self, image_shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tumor shape using Perlin noise.
        
        Args:
            image_shape: Shape of the target image
            
        Returns:
            Tuple of (tumor_probability, tumor_mask)
        """
        
        # Generate Perlin noise-based tumor
        percentile = np.random.uniform(self.mask_percentile_min, self.mask_percentile_max)
        
        try:
            tumor_prob, tumor_mask = generate_shape_3d(
                image_shape, 
                self.perlin_res, 
                percentile, 
                self.device
            )
        except Exception as e:
            print(f"Warning: Perlin noise generation failed: {e}, using simple tumor")
            return self._generate_simple_tumor(image_shape)
        
        # Apply fluid dynamics if enabled
        if self.use_fluid_dynamics:
            tumor_prob = self._apply_fluid_dynamics(tumor_prob)
        
        # Apply size scaling
        tumor_size_factor = random.uniform(*self.tumor_size_factor_range)
        tumor_prob = tumor_prob * tumor_size_factor
        tumor_prob = torch.clamp(tumor_prob, 0, 1)
        
        return tumor_prob, tumor_mask

    
    def _apply_fluid_dynamics(self, tumor_prob: torch.Tensor) -> torch.Tensor:
        if not self.use_fluid_dynamics:
            return tumor_prob

        try:
            nt = np.random.randint(self.min_nt, self.max_nt + 1)

            self.adv_pde.V_dict = generate_velocity_3d(
                tumor_prob.shape,
                self.perlin_res,
                self.V_multiplier,
                self.device,
            )
            print("Generated velocity dict keys:", self.adv_pde.V_dict.keys())

            V_dict = self.adv_pde.V_dict

            if "V" not in V_dict:
                Vx = V_dict.get("Vx")
                Vy = V_dict.get("Vy")
                Vz = V_dict.get("Vz")
                if Vx is None or Vy is None or Vz is None:
                    print("Warning: Velocity components missing, skipping fluid dynamics.")
                    return tumor_prob
                V = torch.stack([Vx, Vy, Vz], dim=-1)  # [Z,Y,X,3]
                self.adv_pde.V_dict["V"] = V
            else:
                V = V_dict["V"]

            if V.shape[:3] != tumor_prob.shape:
                import torch.nn.functional as F
                V_up = F.interpolate(
                    V.permute(3, 0, 1, 2).unsqueeze(0),
                    size=tumor_prob.shape,
                    mode="trilinear",
                    align_corners=False,
                )
                V = V_up.squeeze(0).permute(1, 2, 3, 0)
                self.adv_pde.V_dict["V"] = V

            tumor_prob = odeint(
                self.adv_pde,
                tumor_prob[None],
                self.t[:nt],
                self.dt,
                method=self.integ_method,
            )[-1, 0]

        except Exception as e:
            print(f"Warning: Fluid dynamics failed: {e}, using original shape")
            import traceback
            traceback.print_exc()

        return tumor_prob


class TumorSimulationModule(MapTransform):
    """
    Complete tumor simulation module that integrates age-based segmentation,
    modality-specific intensities, and realistic tumor shape generation.
    """
    
    def __init__(
        self,
        keys: Union[str, List[str]] = "image",
        *,
        device: torch.device = torch.device("cpu"),
        prob: float = 0.3,
        # Age-based segmentation parameters
        use_age_based_segmentation: bool = True,
        segmentation_paths: Optional[Dict[str, str]] = None,
        age_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        # Tumor generation parameters
        perlin_res: List[int] = [2, 2, 2],
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
        # Brain mask parameters (fallback)
        brain_threshold: float = 0.1,
        **unused,
    ):
        super().__init__(keys)
        self.device = device
        self.prob = prob
        self.modality = modality
        self.intensity_variation = intensity_variation
        self.brain_threshold = brain_threshold
        
        # Initialize age-based segmentation loader
        self.use_age_based_segmentation = use_age_based_segmentation
        if self.use_age_based_segmentation:
            if not segmentation_paths or not age_ranges:
                raise ValueError("segmentation_paths and age_ranges must be provided when use_age_based_segmentation=True")
            
            self.seg_loader = AgeBasedSegmentationLoader(
                segmentation_paths=segmentation_paths,
                age_ranges=age_ranges,
                device=device
            )
            print(f"Initialized age-based segmentation with groups: {self.seg_loader.get_available_age_groups()}")
        else:
            self.seg_loader = None
        
        # Initialize tumor shape generator
        self.shape_generator = TumorShapeGenerator(
            device=device,
            perlin_res=perlin_res,
            mask_percentile_min=mask_percentile_min,
            mask_percentile_max=mask_percentile_max,
            tumor_size_factor_range=tumor_size_factor_range,
            pathol_thres=pathol_thres,
            min_tumor_size=min_tumor_size,
            use_fluid_dynamics=use_fluid_dynamics,
            V_multiplier=V_multiplier,
            dt=dt,
            min_nt=min_nt,
            max_nt=max_nt,
            integ_method=integ_method,
            bc=bc,
        )
        
        # Initialize intensity manager
        self.intensity_manager = TumorIntensityManager()
        
        print(f"TumorSimulationModule initialized with probability: {prob}")
        print(f"Age-based segmentation: {use_age_based_segmentation}")
        print(f"Fluid dynamics: {use_fluid_dynamics}")
    
    def _get_brain_mask_from_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """Generate brain mask from segmentation (GM + WM)"""
        # Brain tissue = Gray Matter (1) + White Matter (2)
        # Adjust these values based on your segmentation labels
        brain_mask = ((segmentation == 1) | (segmentation == 2)).float()
        return brain_mask
    
    def _get_brain_mask_from_intensity(self, image: torch.Tensor) -> torch.Tensor:
        """Generate brain mask from image intensity (fallback method)"""
        brain_mask = (image > self.brain_threshold).float()
        return brain_mask
    
    def _apply_tumor_to_image(
        self, 
        image: torch.Tensor, 
        tumor_prob: torch.Tensor, 
        modality: str
    ) -> torch.Tensor:
        """Apply tumor pathology to the image"""
        # Calculate reference intensity
        brain_mask = (image > 0)
        if brain_mask.sum() > 0:
            ref_intensity = (image * brain_mask).sum() / brain_mask.sum()
        else:
            ref_intensity = torch.tensor(1.0, device=self.device)
        
        # Get tumor intensity based on modality
        intensity_multiplier = self.intensity_manager.get_tumor_intensity(
            modality, ref_intensity, self.device
        )
        
        # Add spatial variation to intensity
        intensity_variation = 1.0 + self.intensity_variation * (torch.rand_like(tumor_prob) - 0.5)
        pathol_intensity = ref_intensity * intensity_multiplier * intensity_variation
        
        # Apply pathology based on modality
        modality_upper = modality.upper()
        if modality_upper in ['T2', 'FLAIR']:
            # Hyperintense lesion (additive)
            diseased_image = image + tumor_prob * pathol_intensity
        else:  # T1, T1C
            # Hypointense or enhanced lesion
            if modality_upper == 'T1':
                # Hypointense: reduce original signal and add tumor signal
                diseased_image = image * (1 - tumor_prob * 0.6) + tumor_prob * pathol_intensity
            else:  # T1C
                # Enhanced: additive
                diseased_image = image + tumor_prob * pathol_intensity
        
        # Ensure non-negative values
        diseased_image = torch.clamp(diseased_image, min=0)
        
        return diseased_image
    
    def _generate_tumor_on_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate tumor on a single sample"""
        # Get image
        key = self.keys[0] if isinstance(self.keys, (list, tuple)) else self.keys           
        image = sample[key]
        
        # Ensure image is on correct device
        if image.device != self.device:
            image = image.to(self.device)
        
        # Handle channel dimension
        if image.dim() == 4 and image.shape[0] == 1:
            image_3d = image.squeeze(0)
            add_channel_dim = True
        else:
            image_3d = image
            add_channel_dim = False
        
        # Get age and modality
        age = None
        if 'age' in sample:
            age_tensor = sample['age']
            age = age_tensor.item() if isinstance(age_tensor, torch.Tensor) else float(age_tensor)
        
        modality = sample.get('modality', self.modality)
        
        # Generate tumor shape
        tumor_prob, _ = self.shape_generator.generate_tumor_shape(image_3d.shape)
        
        # Get brain mask
        if self.use_age_based_segmentation and self.seg_loader is not None and age is not None:
            try:
                segmentation = self.seg_loader.get_segmentation_for_age(age)
                brain_mask = self._get_brain_mask_from_segmentation(segmentation)
            except Exception as e:
                print(f"Warning: Failed to use age-based segmentation: {e}, using intensity-based mask")
                brain_mask = self._get_brain_mask_from_intensity(image_3d)
        else:
            brain_mask = self._get_brain_mask_from_intensity(image_3d)
        
        # Restrict tumor to brain tissue
        tumor_prob = tumor_prob * brain_mask
        
        # Check minimum tumor size
        if tumor_prob.sum() < self.shape_generator.min_tumor_size:
            # Try to enlarge tumor
            tumor_prob = tumor_prob * 2.0
            tumor_prob = torch.clamp(tumor_prob, 0, 1)
            tumor_prob = tumor_prob * brain_mask
        
        # Create final tumor mask
        tumor_mask = (tumor_prob > self.shape_generator.pathol_thres).float()
        
        # Apply tumor to image
        diseased_image = self._apply_tumor_to_image(image_3d, tumor_prob, modality)
        
        # Add channel dimension back if needed
        if add_channel_dim:
            diseased_image = diseased_image.unsqueeze(0)
            tumor_mask = tumor_mask.unsqueeze(0)
            tumor_prob = tumor_prob.unsqueeze(0)
        
        # Update sample
        result = dict(sample)
        result[key] = diseased_image
        result['tumor_mask'] = tumor_mask
        result['tumor_prob'] = tumor_prob
        result['has_tumor'] = torch.tensor(True, dtype=torch.bool, device=self.device)
        result['tumor_modality'] = modality
        if age is not None:
            result['tumor_age_group'] = self._get_age_group(age) if self.seg_loader else 'unknown'
        
        return result
    
    def _get_age_group(self, age: float) -> str:
        """Get age group name for given age"""
        if not self.seg_loader:
            return 'unknown'
        
        for age_group, (min_age, max_age) in self.seg_loader.age_ranges.items():
            if min_age <= age < max_age:
                return age_group
        return 'unknown'
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply tumor simulation to the data sample.
        
        Args:
            data: Dictionary containing at least the image key and optionally age, modality
            
        Returns:
            Modified data dictionary with tumor applied (if probability allows)
        """
        # Check if we should apply tumor simulation
        if random.random() >= self.prob:
            # Add metadata indicating no tumor was applied
            result = dict(data)
            result['has_tumor'] = torch.tensor(False, dtype=torch.bool, device=self.device)
            return result
        
        try:
            return self._generate_tumor_on_sample(data)
        except Exception as e:
            print(f"Warning: Tumor generation failed: {e}")
            result = dict(data)
            result['has_tumor'] = torch.tensor(False, dtype=torch.bool, device=self.device)
            return result


# Convenience function for easy integration
def create_tumor_simulator(config: Dict, device: torch.device) -> TumorSimulationModule:
    """
    Create a tumor simulator from configuration dictionary.
    
    Args:
        config: Configuration dictionary with tumor simulation parameters
        device: Device to run simulation on
        
    Returns:
        Configured TumorSimulationModule
    """
    return TumorSimulationModule(device=device, **config)
