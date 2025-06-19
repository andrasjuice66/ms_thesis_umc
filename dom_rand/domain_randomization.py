"""
GPU-aware domain-randomisation pipeline for 3-D brain MR images.

• Fast, GPU-ready MONAI transforms + optional heavy TorchIO artefacts
• All probabilities are configurable (override via `transform_probs`)
• Can be instantiated once and reused safely across workers

Typical use
-----------
dr = DomainRandomizer(device=torch.device("cuda"), **cfg["domain_randomization"])
train_ds = BADataset(..., transform=dr, mode="train")
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, Union, List

import numpy as np
import torch
import torchio as tio
from monai.transforms import (
    Compose,
    RandAffined,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandRicianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandHistogramShiftd,
    RandGibbsNoised,
    RandCoarseDropoutd,
    RandSpatialCropd,
    ToTensord,
    LoadImaged,
    EnsureChannelFirstd,
)

# custom project transforms
from brain_age_pred.dom_rand.custom_transformations import (
    RandomResolutionD,
    RandGammaD,
)

# Import tumor simulator
from brain_age_pred.dom_rand.tumor_simulation import TumorSimulationModule

class DomainRandomizer:
    """
    Random geometric, intensity and artefact transforms for 3-D MRI volumes.
    """

    _DEFAULT_PROBS: Dict[str, float] = {
        # geometric
        "flip"      : 0.5,
        "affine"    : 0.8,
        # intensity
        "contrast"  : 0.6,
        "gamma"     : 0.5,
        "blur"      : 0.4,
        "bias"      : 0.5,
        "scale_int" : 0.4,
        "shift_int" : 0.4,
        "hist_shift": 0.3,
        "noise"     : 0.4,
        "rician"    : 0.3,
        "gibbs"     : 0.3,
        # resolution / dropout
        "resolution": 0.5,
        "coarse_do" : 0.3,
        # heavy artefacts

        # misc
        "crop"      : 1.0,
        # tumor
        "tumor"     : 0.3,
    }
    
    # Default values for transform parameters
    _DEFAULT_PARAMS = {
        # geometric ranges
        "scaling_range": (0.9, 1.1),
        "rotation_range": 10.0,  # degrees
        "shearing_bounds": 0.2,
        # intensity
        "contrast_range": (0.6, 1.4),
        "log_gamma_std": 0.2,
        "bias_field_range": (0.0, 0.6),
        # noise
        "noise_mean": 0.0,
        "noise_std": 0.05,
        "rician_std": 0.05,
        "gibbs_alpha": (0.0, 1.0),
        # blur
        "blur_sigma": (0.5, 1.5),
        # intensity shift
        "shift_offset": (-0.1, 0.1),
        "hist_control_points": (5, 10),
        # resolution
        "max_res_iso": 3.0,
        "min_res": 1.0,
        # dropout
        "coarse_dropout_size": (20, 20, 20),
        "coarse_dropout_holes": 8,
        # spatial crop
        "output_shape": (160, 192, 160),
        "random_center": True,
        # torchio
        "elastic_control_points": 7,
        "elastic_max_displacement": 5.0,
        "spike_num": (1, 6),
        "spike_intensity": (0.1, 0.6),
        "ghost_num": (2, 10),
        # progressive randomization
        "progressive_epochs": 50,
        "progressive_start": 0.2,
        "augmentation_strength": 0.5,
    }

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        device=torch.device("cuda"), 
        image_key: str = "image",
        use_domain_randomization: bool = True,
        # probability overrides
        transform_probs: Optional[Dict[str, float]] = None,
        # Add new parameters for progressive randomization
        progressive_epochs: Optional[int] = None,  
        progressive_start: Optional[float] = None,  
        # geometric ranges
        scaling_range: Optional[Union[Tuple[float, float], List[float]]] = None,
        rotation_range: Optional[float] = None,
        shearing_bounds: Optional[float] = None,
        # intensity
        contrast_range: Optional[Union[Tuple[float, float], List[float]]] = None,
        log_gamma_std: Optional[float] = None,
        bias_field_range: Optional[Union[Tuple[float, float], List[float]]] = None,
        # noise
        noise_mean: Optional[float] = None,
        noise_std: Optional[float] = None,
        rician_std: Optional[float] = None,
        gibbs_alpha: Optional[Union[Tuple[float, float], List[float]]] = None,
        # blur
        blur_sigma: Optional[Union[Tuple[float, float], List[float]]] = None,
        # intensity shift
        shift_offset: Optional[Union[Tuple[float, float], List[float]]] = None,
        hist_control_points: Optional[Union[Tuple[int, int], List[int]]] = None,
        # resolution
        max_res_iso: Optional[float] = None,
        min_res: Optional[float] = None,
        # dropout
        coarse_dropout_size: Optional[Union[Tuple[int, int, int], List[int]]] = None,
        coarse_dropout_holes: Optional[int] = None,
        # spatial crop
        output_shape: Optional[Union[Tuple[int, int, int], List[int]]] = None,
        random_center: Optional[bool] = None,
        # torchio
        use_torchio: bool = False,
        elastic_control_points: Optional[int] = None,
        elastic_max_displacement: Optional[float] = None,
        spike_num: Optional[Union[Tuple[int, int], List[int]]] = None,
        spike_intensity: Optional[Union[Tuple[float, float], List[float]]] = None,
        ghost_num: Optional[Union[Tuple[int, int], List[int]]] = None,
        # Extra params handling
        use_tumor_simulation: Optional[bool] = None,
        tumor_config: Optional[Dict] = None,
        **unused,
    ):
        self.image_key = image_key
        self.use_domain_randomization = use_domain_randomization
        # Convert string device to torch.device object if needed
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_tio = use_torchio
        
        # Store tumor-related parameters
        self.use_tumor_simulation = use_tumor_simulation
        self.tumor_config = tumor_config if tumor_config else {}
        self.tumor_simulator = None
        
        # Initialize parameters with defaults, then override with provided values
        params = {**self._DEFAULT_PARAMS}
        
        # Override default parameters with provided values if not None
        # Progressive randomization parameters
        if progressive_epochs is not None:
            params["progressive_epochs"] = progressive_epochs
        if progressive_start is not None:
            params["progressive_start"] = progressive_start
            
        # Geometric parameters
        if scaling_range is not None:
            params["scaling_range"] = tuple(scaling_range) if isinstance(scaling_range, list) else scaling_range
        if rotation_range is not None:
            params["rotation_range"] = rotation_range
        if shearing_bounds is not None:
            params["shearing_bounds"] = shearing_bounds
            
        # Intensity parameters
        if contrast_range is not None:
            params["contrast_range"] = tuple(contrast_range) if isinstance(contrast_range, list) else contrast_range
        if log_gamma_std is not None:
            params["log_gamma_std"] = log_gamma_std
        if bias_field_range is not None:
            params["bias_field_range"] = tuple(bias_field_range) if isinstance(bias_field_range, list) else bias_field_range
            
        # Noise parameters
        if noise_mean is not None:
            params["noise_mean"] = noise_mean
        if noise_std is not None:
            params["noise_std"] = noise_std
        if rician_std is not None:
            params["rician_std"] = rician_std
        if gibbs_alpha is not None:
            params["gibbs_alpha"] = tuple(gibbs_alpha) if isinstance(gibbs_alpha, list) else gibbs_alpha
            
        # Blur parameters
        if blur_sigma is not None:
            params["blur_sigma"] = tuple(blur_sigma) if isinstance(blur_sigma, list) else blur_sigma
            
        # Intensity shift parameters
        if shift_offset is not None:
            params["shift_offset"] = tuple(shift_offset) if isinstance(shift_offset, list) else shift_offset
        if hist_control_points is not None:
            params["hist_control_points"] = tuple(hist_control_points) if isinstance(hist_control_points, list) else hist_control_points
            
        # Resolution parameters
        if max_res_iso is not None:
            params["max_res_iso"] = max_res_iso
        if min_res is not None:
            params["min_res"] = min_res
            
        # Dropout parameters
        if coarse_dropout_size is not None:
            params["coarse_dropout_size"] = tuple(coarse_dropout_size) if isinstance(coarse_dropout_size, list) else coarse_dropout_size
        if coarse_dropout_holes is not None:
            params["coarse_dropout_holes"] = coarse_dropout_holes
            
        # Spatial crop parameters
        if output_shape is not None:
            params["output_shape"] = tuple(output_shape) if isinstance(output_shape, list) else output_shape
        if random_center is not None:
            params["random_center"] = random_center
            
        # TorchIO parameters
        if elastic_control_points is not None:
            params["elastic_control_points"] = elastic_control_points
        if elastic_max_displacement is not None:
            params["elastic_max_displacement"] = elastic_max_displacement
        if spike_num is not None:
            params["spike_num"] = tuple(spike_num) if isinstance(spike_num, list) else spike_num
        if spike_intensity is not None:
            params["spike_intensity"] = tuple(spike_intensity) if isinstance(spike_intensity, list) else spike_intensity
        if ghost_num is not None:
            params["ghost_num"] = tuple(ghost_num) if isinstance(ghost_num, list) else ghost_num
            
        # Store all parameters as instance attributes
        self.scaling_range = params["scaling_range"]
        self.rotation_range = params["rotation_range"]
        self.shearing_bounds = params["shearing_bounds"]
        self.contrast_range = params["contrast_range"]
        self.log_gamma_std = params["log_gamma_std"]
        self.bias_field_rng = params["bias_field_range"]
        self.noise_mean = params["noise_mean"]
        self.noise_std = params["noise_std"]
        self.rician_std = params["rician_std"]
        self.gibbs_alpha = params["gibbs_alpha"]
        self.blur_sigma = params["blur_sigma"]
        self.shift_offset = params["shift_offset"]
        self.hist_control_points = params["hist_control_points"]
        self.max_res_iso = params["max_res_iso"]
        self.min_res = params["min_res"]
        self.coarse_size = params["coarse_dropout_size"]
        self.coarse_holes = params["coarse_dropout_holes"]
        self.output_shape = params["output_shape"]
        self.random_center = params["random_center"]
        self.elastic_control_points = params["elastic_control_points"]
        self.elastic_max_displacement = params["elastic_max_displacement"]
        self.spike_num = params["spike_num"]
        self.spike_intensity = params["spike_intensity"]
        self.ghost_num = params["ghost_num"]
        self.progressive_epochs = params["progressive_epochs"]
        self.progressive_start = params["progressive_start"]
        self.augmentation_strength = params["augmentation_strength"]

        # merge / override probabilities
        self.prob = {**DomainRandomizer._DEFAULT_PROBS}
        if transform_probs:
            self.prob.update(transform_probs)

        # Store original probabilities for reference
        self.original_probs = {**self.prob}
        
        # Initialize with starting probabilities
        self.current_epoch = 0
        self._update_progressive_probs()

        # build transform pipelines
        self._build_monai_pipeline()
        self._build_torchio_pipeline()
        self._setup_tumor_simulator()


    # ------------------------------------------------------------------ #
    #                          MONAI pipeline                             #
    # ------------------------------------------------------------------ #
    def _build_monai_pipeline(self) -> None:
        deg2rad = np.pi / 180
        tfms = []

        # 1. flips & affine
        tfms.extend([
            RandFlipd(
                keys=[self.image_key],
                prob=self.prob["flip"],
                spatial_axis=(0, 1, 2),
            ),
            RandAffined(
                keys=[self.image_key],
                prob=self.prob["affine"],
                rotate_range=(deg2rad * self.rotation_range,) * 3,
                scale_range=(self.scaling_range[1] - 1,) * 3,
                shear_range=(self.shearing_bounds,) * 3,
                mode="bilinear",
            ),
        ])

        # 2. basic intensity
        tfms.extend([
            RandAdjustContrastd(
                keys=[self.image_key],
                prob=self.prob["contrast"],
                gamma=self.contrast_range,
            ),
            RandGammaD(
                keys=[self.image_key],
                log_gamma_std=self.log_gamma_std,
                prob=self.prob["gamma"],
            ),
            RandScaleIntensityd(
                keys=[self.image_key],
                prob=self.prob["scale_int"],
                factors=self.contrast_range,
            ),
            RandShiftIntensityd(
                keys=[self.image_key],
                prob=self.prob["shift_int"],
                offsets=self.shift_offset,
            ),
            RandHistogramShiftd(
                keys=[self.image_key],
                prob=self.prob["hist_shift"],
                num_control_points=self.hist_control_points,
            ),
        ])

        # 3. noise / artefacts
        tfms.extend([
            RandGaussianNoised(
                keys=[self.image_key],
                prob=self.prob["noise"],
                mean=self.noise_mean,
                std=self.noise_std,
            ),
            RandRicianNoised(
                keys=[self.image_key],
                prob=self.prob["rician"],
                std=self.rician_std,
            ),
            RandGibbsNoised(
                keys=[self.image_key],
                prob=self.prob["gibbs"],
                alpha=self.gibbs_alpha,
            ),
            RandGaussianSmoothd(
                keys=[self.image_key],
                prob=self.prob["blur"],
                sigma_x=self.blur_sigma,
                sigma_y=self.blur_sigma,
                sigma_z=self.blur_sigma,
            ),
            RandBiasFieldd(
                keys=[self.image_key],
                prob=self.prob["bias"],
                coeff_range=self.bias_field_rng,
            ),
            RandomResolutionD(
                keys=[self.image_key],
                min_res=self.min_res,
                max_res_iso=self.max_res_iso,
                prob=self.prob["resolution"],
            ),
            RandCoarseDropoutd(
                keys=[self.image_key],
                prob=self.prob["coarse_do"],
                holes=self.coarse_holes,
                spatial_size=self.coarse_size,
                fill_value=0.0,
            ),
        ])

        # # 4. optional crop to ROI
        # if self.output_shape is not None:
        #     tfms.append(
        #         RandSpatialCropd(
        #             keys=[self.image_key],
        #             roi_size=self.output_shape,
        #             random_center=self.random_center,
        #             random_size=False,
        #         )
        #     )

        # 5. tensor conversion
        tfms.append(ToTensord(keys=[self.image_key]))

        # compose & push to GPU if possible
        self.monai = Compose(tfms)
        if self.device.type == "cuda": #I was getting OOM errors when pushing to GPU
            for t in self.monai.transforms:
                if hasattr(t, "set_device"):
                    t.set_device(self.device)

   
    def _build_torchio_pipeline(self) -> None:
        if not self.use_tio:
            self.tio = None
            return

        self.tio = tio.Compose([
            tio.RandomElasticDeformation(
                num_control_points=self.elastic_control_points,
                max_displacement=self.elastic_max_displacement,
                locked_borders=2,
                p=self.prob["elastic"],
            ),
            tio.RandomSpike(
                num_spikes=self.spike_num,
                intensity=self.spike_intensity,
                p=self.prob["spike"],
            ),
            tio.RandomGhosting(
                num_ghosts=self.ghost_num,
                axes=(0, 1, 2),
                p=self.prob["ghost"],
            ),
        ])

    def _update_progressive_probs(self) -> None:
        """Update transform probabilities based on current epoch"""
        if self.progressive_epochs <= 0:
            return
            
        progress = min(1.0, self.current_epoch / self.progressive_epochs)
        
        for key in self.original_probs:
            start_prob = self.original_probs[key] * self.progressive_start
            final_prob = self.original_probs[key] * self.augmentation_strength
            self.prob[key] = start_prob + (final_prob - start_prob) * progress

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, epoch: int) -> None:
        """Update current epoch and adjust probabilities accordingly."""
        self._current_epoch = epoch
        self._update_progressive_probs()

    def _setup_tumor_simulator(self) -> None:
        """Initialize the tumor simulator if enabled"""
        if not self.use_tumor_simulation:
            return
        
        try:
            tumor_prob = self.prob.get("tumor", 0.3)
            if not self.tumor_config:
                print("Warning: No tumor configuration provided, using defaults")
            
            self.tumor_simulator = TumorSimulationModule(
                keys=self.image_key,
                device=self.device,
                prob=tumor_prob,
                **self.tumor_config or {}
            )
            print(f"Tumor simulator initialized with probability: {tumor_prob}")
            if self.tumor_config:
                print(f"Tumor configuration: {self.tumor_config}")
        except Exception as e:
            print(f"Warning: Failed to initialize tumor simulator: {str(e)}")
            print("Tumor simulation will be disabled")
            self.tumor_simulator = None
            self.use_tumor_simulation = False

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply domain randomization and tumor simulation to sample"""
        if sample is None:
            raise RuntimeError("Input sample is None")
        
        if self.image_key not in sample:
            raise RuntimeError(f"Image key '{self.image_key}' not found in sample")
        
        if sample[self.image_key] is None:
            raise RuntimeError("Image is None in sample")

        # Apply domain randomization transforms
        if self.use_domain_randomization and self.monai is not None:
            img = sample[self.image_key]
            transform_input = {self.image_key: img}
            result = self.monai(transform_input)
            
            # Check if result is None or if image key is missing
            if result is None:
                raise RuntimeError("DomainRandomizer: MONAI pipeline returned None")
            
            if self.image_key not in result:
                raise RuntimeError(f"DomainRandomizer: Image key '{self.image_key}' missing after transforms")
            
            img = result[self.image_key]
            if img is None:
                raise RuntimeError(f"DomainRandomizer: Image is None after transforms")

            # Update sample with transformed image
            sample[self.image_key] = img
        
        # Apply tumor simulation
        if self.use_tumor_simulation and self.tumor_simulator is not None:
            try:
                sample = self.tumor_simulator(sample)
            except Exception as e:
                print(f"Warning: Tumor simulation failed: {e}")
                sample['has_tumor'] = torch.tensor(False, dtype=torch.bool)

        # Ensure tensor consistency
        img = sample[self.image_key]
        for k in ("age", "weight"):
            if k in sample and sample[k].device != img.device:
                sample[k] = sample[k].to(img.device)

        return sample

