from __future__ import annotations
from typing import Dict, Tuple, Optional, Union, List

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
    RandZoomd,
    RandRotated,
    ToTensord,
)

# custom project transforms
from brain_age_pred.dataset.custom_transformations import (
    RandomResolutionD,
    RandGammaD,
)


class AugmentationPipeline:
    """
    Random geometric and non-spatial transforms for 3-D MRI volumes.

    Switches:
      - use_spatial_transforms: controls only flips/affine
      - use_intensity_transforms: controls everything else (intensity, noise/artifacts,
        resolution, dropout, and TorchIO transforms)
    """

    _DEFAULT_PROBS: Dict[str, float] = {
        # spatial
        "flip": 0.5,
        "affine": 0.5,
        "zoom": 0.5,  # Added for RandZoomd
        "rotate": 0.5,  # Added for RandRotated
        # non-spatial (intensity + artifacts)
        "contrast": 0.5,
        "gamma": 0.5,
        "blur": 0.3,
        "bias": 0.5,  # Updated from 0.3 to 0.5
        "scale_int": 0.4,
        "shift_int": 0.4,
        "hist_shift": 0.3,
        "noise": 0.4,
        "rician": 0.3,
        "gibbs": 0.3,
        # resolution / dropout
        "resolution": 0.5,
        "coarse_do": 0.3,
        # torchio
        "spike": 0.5,
        "ghost": 0.5,
        "motion": 0.5,
        "swap": 0.5,
    }
    
    _DEFAULT_PARAMS = {
        # spatial ranges
        "scaling_range": (0.9, 1.1),
        "rotation_range": 10.0,  # degrees
        "shearing_bounds": 0.2,
        # zoom parameters for RandZoomd
        "zoom_min": 0.95,
        "zoom_max": 1.00,
        # rotation parameters for RandRotated
        "rotate_range_x": 0.1,
        "rotate_range_y": 0.1,
        "rotate_range_z": 0.1,
        # intensity
        "contrast_range": (0.6, 3.0),  # Updated from (0.6, 1.4) to (0.6, 3.0)
        "log_gamma_std": 0.2,
        "bias_field_range": (-0.5, 0.1),  # Updated from (0.0, 0.6) to (-0.5, 0.1)
        "bias_field_degree": 5,  # Added for RandBiasFieldd degree parameter
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
        # spatial crop parameters
        "output_shape": None,
        "random_center": True,
        # torchio params
        "spike_num": (1, 6),
        "spike_intensity": (0.1, 0.6),
        "ghost_num": (1, 4),
        "ghost_intensity": (0.1, 0.6),
        "motion_degrees": 3.0,
        "motion_translation": 5.0,
        "motion_transforms": 4,
        "tio_gamma_log": 0.8,
        "tio_noise_std": (0.0, 0.5),
    }

    def __init__(
        self,
        *,
        device=torch.device("cuda"),
        image_key: str = "image",
        use_spatial_transforms: bool = True,
        use_intensity_transforms: bool = True,  # BIG switch for all non-spatial
        # Augmentation control parameters
        use_augmentation: bool = True,  # Master switch for augmentation
        augmentation_strength: float = 1.0,  # Overall augmentation strength multiplier
        # probability overrides
        transform_probs: Optional[Dict[str, float]] = None,
        # spatial ranges
        scaling_range: Optional[Union[Tuple[float, float], List[float]]] = None,
        rotation_range: Optional[float] = None,
        shearing_bounds: Optional[float] = None,
        # zoom parameters
        zoom_min: Optional[float] = None,
        zoom_max: Optional[float] = None,
        # rotation parameters for RandRotated
        rotate_range_x: Optional[float] = None,
        rotate_range_y: Optional[float] = None,
        rotate_range_z: Optional[float] = None,
        # intensity
        contrast_range: Optional[Union[Tuple[float, float], List[float]]] = None,
        log_gamma_std: Optional[float] = None,
        bias_field_range: Optional[Union[Tuple[float, float], List[float]]] = None,
        bias_field_degree: Optional[int] = None,
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
        # spatial crop parameters
        output_shape: Optional[Union[Tuple[int, int, int], List[int]]] = None,
        random_center: bool = True,
        # torchio parameters
        spike_num: Optional[Union[Tuple[int, int], List[int]]] = None,
        spike_intensity: Optional[Union[Tuple[float, float], List[float]]] = None,
        ghost_num: Optional[Union[Tuple[int, int], List[int]]] = None,
        ghost_intensity: Optional[Union[Tuple[float, float], List[float]]] = None,
        motion_degrees: Optional[float] = None,
        motion_translation: Optional[float] = None,
        motion_transforms: Optional[int] = None,
        tio_gamma_log: Optional[float] = None,
        tio_noise_std: Optional[Union[Tuple[float, float], List[float]]] = None,
        **unused,
    ):
        self.image_key = image_key
        self.use_spatial_transforms = use_spatial_transforms and use_augmentation
        self.use_intensity_transforms = use_intensity_transforms and use_augmentation
        self.use_augmentation = use_augmentation
        self.augmentation_strength = augmentation_strength
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Initialize params with defaults and override with provided values
        params = {**self._DEFAULT_PARAMS}
        param_mapping = {
            'scaling_range': scaling_range,
            'rotation_range': rotation_range,
            'shearing_bounds': shearing_bounds,
            'zoom_min': zoom_min,
            'zoom_max': zoom_max,
            'rotate_range_x': rotate_range_x,
            'rotate_range_y': rotate_range_y,
            'rotate_range_z': rotate_range_z,
            'contrast_range': contrast_range,
            'log_gamma_std': log_gamma_std,
            'bias_field_range': bias_field_range,
            'bias_field_degree': bias_field_degree,
            'noise_mean': noise_mean,
            'noise_std': noise_std,
            'rician_std': rician_std,
            'gibbs_alpha': gibbs_alpha,
            'blur_sigma': blur_sigma,
            'shift_offset': shift_offset,
            'hist_control_points': hist_control_points,
            'max_res_iso': max_res_iso,
            'min_res': min_res,
            'coarse_dropout_size': coarse_dropout_size,
            'coarse_dropout_holes': coarse_dropout_holes,
            'output_shape': output_shape,
            'random_center': random_center,
            'spike_num': spike_num,
            'spike_intensity': spike_intensity,
            'ghost_num': ghost_num,
            'ghost_intensity': ghost_intensity,
            'motion_degrees': motion_degrees,
            'motion_translation': motion_translation,
            'motion_transforms': motion_transforms,
            'tio_gamma_log': tio_gamma_log,
            'tio_noise_std': tio_noise_std,
        }
        for param_name, param_value in param_mapping.items():
            if param_value is not None:
                if isinstance(param_value, list) and param_name in [
                    'scaling_range', 'contrast_range', 'bias_field_range',
                    'gibbs_alpha', 'blur_sigma', 'shift_offset', 'hist_control_points',
                    'coarse_dropout_size', 'spike_num', 'spike_intensity',
                    'ghost_num', 'ghost_intensity', 'tio_noise_std', 'output_shape'
                ]:
                    params[param_name] = tuple(param_value)
                else:
                    params[param_name] = param_value
        for k, v in params.items():
            setattr(self, k, v)

        # probabilities - apply augmentation strength multiplier
        self.prob = {**AugmentationPipeline._DEFAULT_PROBS}
        if transform_probs:
            # Apply augmentation strength to all probabilities
            scaled_probs = {}
            for key, value in transform_probs.items():
                scaled_probs[key] = min(1.0, value * self.augmentation_strength)
            self.prob.update(scaled_probs)

        # build pipelines
        if self.use_augmentation:
            self._build_monai_pipeline()
            if self.use_intensity_transforms:
                self._build_torchio_pipeline()
            else:
                self.tio = None
        else:
            self.monai = None
            self.tio = None

    def _build_monai_pipeline(self) -> None:
        """Build MONAI transformation pipeline."""
        deg2rad = np.pi / 180
        tfms = []

        # Spatial only
        if self.use_spatial_transforms:
            tfms.extend([
                RandFlipd(
                    keys=[self.image_key],
                    prob=self.prob["flip"],
                    spatial_axis=2,
                ),
                RandZoomd(
                    keys=[self.image_key], 
                    min_zoom=self.zoom_min, 
                    max_zoom=self.zoom_max, 
                    prob=self.prob["zoom"]
                ),
                RandRotated(
                    keys=[self.image_key], 
                    range_x=self.rotate_range_x, 
                    range_y=self.rotate_range_y, 
                    range_z=self.rotate_range_z, 
                    prob=self.prob["rotate"]
                ),
                RandAffined(
                    keys=[self.image_key],
                    prob=self.prob["affine"],
                    rotate_range=(0, 0, 0),  # Disable rotation since we use RandRotated
                    scale_range=(0, 0, 0),   # Disable scaling since we use RandZoomd
                    shear_range=(self.shearing_bounds,) * 3,
                    mode="bilinear",
                ),
            ])

        # Everything non-spatial under the single switch
        if self.use_intensity_transforms:
            # Intensity family
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

            # Noise / artefacts
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
                    degree=self.bias_field_degree,
                    coeff_range=self.bias_field_range,
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
                    holes=self.coarse_dropout_holes,
                    spatial_size=self.coarse_dropout_size,
                    fill_value=0.0,
                ),
            ])

        # Always end with tensor conversion to normalize output type
        tfms.append(ToTensord(keys=[self.image_key]))

        self.monai = Compose(tfms)
        if self.device.type == "cuda":
            for t in self.monai.transforms:
                if hasattr(t, "set_device"):
                    t.set_device(self.device)

    def _build_torchio_pipeline(self) -> None:
        """Build TorchIO pipeline (all considered non-spatial under the big switch)."""
        tfms = [
            tio.RandomSpike(
                num_spikes=self.spike_num,
                intensity=self.spike_intensity,
                p=self.prob["spike"],
            ),
            tio.RandomGhosting(
                num_ghosts=self.ghost_num,
                intensity=self.ghost_intensity,
                p=self.prob["ghost"],
            ),
            tio.RandomMotion(
                degrees=self.motion_degrees,
                translation=self.motion_translation,
                num_transforms=self.motion_transforms,
                p=self.prob["motion"],
            ),
            tio.RandomSwap(
                patch_size=15,
                num_iterations=100,
                p=self.prob["swap"],
            ),
        ]
        self.tio = tio.Compose(tfms)

    def _apply_monai(self, img: torch.Tensor) -> torch.Tensor:
        """Apply MONAI transforms with uniform validation."""
        if self.monai is None:
            return img
        try:
            out = self.monai({self.image_key: img})
        except Exception as e:
            raise RuntimeError(f"MONAI transforms failed: {e}") from e

        if out is None or self.image_key not in out:
            raise RuntimeError("MONAI pipeline returned invalid output (missing image key).")
        new_img = out[self.image_key]
        if new_img is None:
            raise RuntimeError("MONAI pipeline produced None image.")
        return new_img

    def _apply_torchio(self, img: torch.Tensor) -> torch.Tensor:
        """Apply TorchIO transforms with device consistency."""
        if not (self.use_intensity_transforms and self.tio is not None):
            return img

        orig_device = img.device
        try:
            subject = tio.Subject(img=tio.ScalarImage(tensor=img))
            transformed = self.tio(subject)
            new_img = transformed.img.data
        except Exception as e:
            raise RuntimeError(f"TorchIO transforms failed: {e}") from e

        if new_img.device != orig_device:
            new_img = new_img.to(orig_device)
        return new_img

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Uniformly apply transforms and keep sample tensors on same device as image."""
        if sample is None:
            raise RuntimeError("Input sample is None")
        if self.image_key not in sample:
            raise RuntimeError(f"Image key '{self.image_key}' not found in sample")
        if sample[self.image_key] is None:
            raise RuntimeError("Image is None in sample")

        img = sample[self.image_key]
        img = self._apply_monai(img)
        img = self._apply_torchio(img)
        sample[self.image_key] = img

        # Keep auxiliary tensors on same device as image
        for k, v in list(sample.items()):
            if k != self.image_key and torch.is_tensor(v) and v.device != img.device:
                sample[k] = v.to(img.device)

        return sample