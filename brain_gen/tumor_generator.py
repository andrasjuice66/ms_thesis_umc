from __future__ import annotations
from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import torch
import random
from monai.transforms.transform import Transform
from monai.transforms import RandomizableTransform
import torch.nn.functional as F
from brain_age_pred.dataset.FluidAnomaly.perlin3d import generate_shape_3d, generate_velocity_3d
from brain_age_pred.dataset.FluidAnomaly.DiffEqs.pde import AdvDiffPDE
from brain_age_pred.dataset.FluidAnomaly.DiffEqs.odeint import odeint


class TumorShapeGenerator:
    """
    Generates tumor shapes using Perlin noise and optional fluid dynamics.
    """
    
    def __init__(
        self,
        device: torch.device,
        perlin_res: List[int] = [4, 4, 4],
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
            self.t = torch.arange(max_nt, dtype=torch.float64, device=device) * dt
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

    def _generate_simple_tumor(self, image_shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a simple ellipsoidal tumor as fallback"""
        # Create a simple ellipsoidal tumor in the center
        z, y, x = image_shape
        center = (z // 2, y // 2, x // 2)
        
        # Random ellipsoid parameters
        a = random.uniform(5, 15)  # semi-axis lengths
        b = random.uniform(5, 15)
        c = random.uniform(5, 15)
        
        # Create coordinate grids
        zz, yy, xx = torch.meshgrid(
            torch.arange(z, device=self.device),
            torch.arange(y, device=self.device),
            torch.arange(x, device=self.device),
            indexing='ij'
        )
        
        # Ellipsoid equation
        tumor_prob = torch.zeros_like(zz, dtype=torch.float32)
        ellipsoid = ((zz - center[0])**2 / a**2 + 
                     (yy - center[1])**2 / b**2 + 
                     (xx - center[2])**2 / c**2)
        tumor_prob[ellipsoid <= 1] = 0.8
        
        tumor_mask = tumor_prob > 0.5
        return tumor_prob, tumor_mask.float()
    
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
                V_up = F.interpolate(
                    V.permute(3, 0, 1, 2).unsqueeze(0),
                    size=tumor_prob.shape,
                    mode="trilinear",
                    align_corners=False,
                )
                V = V_up.squeeze(0).permute(1, 2, 3, 0)
                self.adv_pde.V_dict["V"] = V

            result = odeint(
                self.adv_pde,
                tumor_prob.double().unsqueeze(0),
                self.t[:nt],
                self.dt,
                method=self.integ_method,
            )[-1, 0].float()
            
            # Validate fluid dynamics result
            if torch.isnan(result).any() or torch.isinf(result).any():
                print("Warning: Fluid dynamics produced NaN/Inf values, using original shape")
                return tumor_prob
            
            # Clamp to valid probability range
            result = torch.clamp(result, 0, 1)
            return result

        except Exception as e:
            print(f"Warning: Fluid dynamics failed: {e}, using original shape")
            return tumor_prob


class TumorSampleConditionalGMMd(Transform):
    """
    Tumor generation transform that integrates with brain generator pipeline.
    Generates tumor shapes using Perlin noise and samples intensities using the same 
    prior distributions as brain tissues.
    """
    
    def __init__(
        self,
        seg_key: str = "seg_gt",
        image_key: str = "image",
        prior_means: np.ndarray = None,
        prior_stds: np.ndarray = None,
        distribution: str = "uniform",
        # Shape generation parameters
        perlin_res: List[int] = [4, 4, 4],
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
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            seg_key: Key for segmentation data
            image_key: Key for image data
            prior_means: Same prior means used for brain tissue sampling
            prior_stds: Same prior stds used for brain tissue sampling
            distribution: "normal" or "uniform" for sampling (should match brain tissue)
            Other parameters: Same as TumorShapeGenerator
        """
        self.seg_key = seg_key
        self.image_key = image_key
        self.distribution = distribution
        self.device = device
        
        # Use the same priors as brain tissues
        if prior_means is None:
            raise ValueError("prior_means must be provided (same as brain tissue sampling)")
        if prior_stds is None:
            raise ValueError("prior_stds must be provided (same as brain tissue sampling)")
        
        self.prior_means = np.asarray(prior_means, dtype=float)
        self.prior_stds = np.asarray(prior_stds, dtype=float)
        
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
    
    def __call__(self, data):
        """Apply tumor generation to the data"""
        # Get segmentation and image
        seg = data[self.seg_key]  # shape [1, D, H, W] or [C, D, H, W]
        image = data[self.image_key]  # Already generated brain tissue intensities
        
        # Drop channel-first if it's size 1
        if seg.dim() == 4 and seg.shape[0] == 1:
            seg_3d = seg.squeeze(0)
            image_3d = image.squeeze(0)
            add_channel_dim = True
        else:
            seg_3d = seg
            image_3d = image
            add_channel_dim = False
        
        device = seg.device
        
        # Generate tumor shape
        tumor_mask, tumor_prob = self.shape_generator.generate_tumor_shape(seg_3d.shape)
        
        # Create brain mask from segmentation (GM=1, WM=2)
        brain_mask = ((seg_3d == 1) | (seg_3d == 2)).float()
        
        # Restrict tumor to brain regions
        tumor_prob = tumor_prob * brain_mask
        
        # Check minimum tumor size
        if tumor_prob.sum() < self.shape_generator.min_tumor_size:
            # Try to enlarge tumor
            tumor_prob = tumor_prob * 2.0
            tumor_prob = torch.clamp(tumor_prob, 0, 1)
            tumor_prob = tumor_prob * brain_mask
        
        # Create final tumor mask
        final_tumor_mask = (tumor_prob > self.shape_generator.pathol_thres).float()
        
        # Sample tumor intensity using the same approach as brain tissues
        # Use the same prior distributions
        loc_means = torch.tensor(self.prior_means[0], device=device, dtype=torch.float32)
        scale_means = torch.tensor(self.prior_means[1], device=device, dtype=torch.float32)
        loc_stds = torch.tensor(self.prior_stds[0], device=device, dtype=torch.float32)
        scale_stds = torch.tensor(self.prior_stds[1], device=device, dtype=torch.float32)
        
        # Pick a random tissue class for tumor intensity (excluding background)
        n_classes = loc_means.shape[0]
        # Ensure we have classes to choose from besides background
        if n_classes > 1:
            class_idx = random.randint(1, n_classes - 1)
        else:
            class_idx = 0 # Fallback to whatever is available

        if self.distribution == "normal":
            tumor_mean = torch.normal(loc_means[class_idx], scale_means[class_idx])
            tumor_std = torch.normal(loc_stds[class_idx], scale_stds[class_idx])
        else:  # uniform
            low_m = loc_means[class_idx] - scale_means[class_idx]
            high_m = loc_means[class_idx] + scale_means[class_idx]
            tumor_mean = low_m + (high_m - low_m) * torch.rand(1, device=device).squeeze()
            
            low_s = loc_stds[class_idx] - scale_stds[class_idx]
            high_s = loc_stds[class_idx] + scale_stds[class_idx]
            tumor_std = low_s + (high_s - low_s) * torch.rand(1, device=device).squeeze()
        
        # Force positive std
        tumor_std = torch.abs(tumor_std) + 1e-6
        
        # Sample tumor intensities for each tumor voxel
        # Now tumor_mean and tumor_std are scalar tensors, so we don't need .item()
        # and can create a tensor of the desired size.
        tumor_intensities = torch.normal(tumor_mean, tumor_std, size=tumor_prob.shape, device=device)
        
        # Apply tumor to image
        # Blend tumor intensities with original image based on tumor probability
        diseased_image = image_3d * (1 - tumor_prob) + tumor_intensities * tumor_prob
        
        # Add channel dimension back if needed
        if add_channel_dim:
            diseased_image = diseased_image.unsqueeze(0)
            final_tumor_mask = final_tumor_mask.unsqueeze(0)
            tumor_prob = tumor_prob.unsqueeze(0)
        
        # Update the data
        data[self.image_key] = diseased_image
        data['tumor_mask'] = final_tumor_mask
        data['tumor_prob'] = tumor_prob
        data['has_tumor'] = torch.tensor(True, dtype=torch.bool)
        
        return data


class RandTumorSampleConditionalGMMd(RandomizableTransform):
    """
    Randomizable wrapper for tumor generation with probability control.
    """
    
    def __init__(
        self,
        seg_key: str = "seg_gt",
        image_key: str = "image",
        prior_means: np.ndarray = None,
        prior_stds: np.ndarray = None,
        distribution: str = "uniform",
        prob: float = 0.3,
        # Shape generation parameters
        perlin_res: List[int] = [4, 4, 4],
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
        device: torch.device = torch.device("cpu"),
    ):
        RandomizableTransform.__init__(self, prob)
        
        self.tumor_transform = TumorSampleConditionalGMMd(
            seg_key=seg_key,
            image_key=image_key,
            prior_means=prior_means,
            prior_stds=prior_stds,
            distribution=distribution,
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
            device=device,
        )
    
    def __call__(self, data):
        """Apply tumor generation with probability control"""
        self.randomize(data)
        if self._do_transform:
            return self.tumor_transform(data)
        else:
            # Add metadata indicating no tumor was applied
            result = dict(data)
            result['has_tumor'] = torch.tensor(False, dtype=torch.bool)
            return result
