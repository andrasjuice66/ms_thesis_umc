from __future__ import annotations
import numpy as np
from typing import Sequence

import torch
import torchio as tio
from monai.transforms import (
    Compose, RandFlipd, RandAffined, RandAdjustContrastd, RandBiasFieldd,
    RandGaussianSmoothd, RandGaussianNoised, RandRicianNoised, RandGibbsNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandHistogramShiftd, ToTensord,
    RandSpatialCropd, SpatialPadd, CopyItemsd, DeleteItemsd, Transform,
    CenterSpatialCropd, RandZoomd, RandRotated
)

# ------------------------------------------------------------------ #
# project imports
# ------------------------------------------------------------------ #
from brain_age_pred.dataset.custom_transformations import (
    RandomResolutionD, RandGammaD, HemisphereAwareFlipD, DynamicResolutionD,
    IntensityClipNormalizeD, ConvertLabelsD,
)
from brain_age_pred.brain_gen.gen_image_from_labels import (
    SampleConditionalGMMd, MultiChannelSampleConditionalGMMd,
)
from brain_age_pred.brain_gen.tumor_generator import RandTumorSampleConditionalGMMd
from brain_age_pred.brain_gen.labels import (
    GENERATION_LABELS, GENERATION_CLASSES, N_NEUTRAL_LABELS,
)

# ------------------------------------------------------------------ #
# TorchIO Transform that works with MONAI Compose
# ------------------------------------------------------------------ #
class TorchIOTransformd(Transform):
    """Apply TorchIO transforms within MONAI Compose pipeline."""
    
    def __init__(self, keys, transforms):
        self.keys = keys if isinstance(keys, list) else [keys]
        if isinstance(transforms, list):
            self.tio_transform = tio.Compose(transforms)
        else:
            self.tio_transform = transforms
    
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                img = data[key]
                
                # Convert to TorchIO format
                if torch.is_tensor(img):
                    if img.ndim == 3:
                        img = img.unsqueeze(0)
                    subject = tio.Subject(img=tio.ScalarImage(tensor=img))
                else:
                    subject = tio.Subject(img=tio.ScalarImage(tensor=torch.tensor(img)))
                
                # Apply TorchIO transforms
                transformed = self.tio_transform(subject)
                
                # Extract transformed image
                transformed_img = transformed.img.data
                
                # Maintain original shape if needed
                if data[key].ndim == 3 and transformed_img.ndim == 4:
                    transformed_img = transformed_img.squeeze(0)
                
                data[key] = transformed_img
        
        return data


# ------------------------------------------------------------------ #
# cheap in-place background mask
# ------------------------------------------------------------------ #
class ZeroBackgroundd(Transform):
    """img *= (seg_gt != 0) — in-place, no extra image allocation."""
    def __init__(self, img_key="image", seg_key="seg_gt"):
        self.img_key, self.seg_key = img_key, seg_key

    def __call__(self, data):
        data[self.img_key].mul_(data[self.seg_key].ne(0))
        return data


# ================================================================== #
#                           brain generator
# ================================================================== #
class BABrainGenerator:
    # -------------------------------------------------------------- #
    def __init__(
        self,
        prior_means: np.ndarray,
        prior_stds: np.ndarray,
        distribution: str,
        prob: dict,
        # spatial
        rotation_range: float,
        scaling_range: float,
        shear_bounds: float,
        translation_bounds: float,
        # intensity
        contrast_range: tuple[float, float],
        log_gamma_std: float,
        shift_offset: float,
        hist_control_points: int,
        # artefacts
        noise_mean: float,
        noise_std: float,
        rician_std: float,
        gibbs_alpha: float,
        blur_sigma: float,
        bias_field_rng: tuple[float, float],
        # resolution
        min_res: float,
        max_res_iso: float,
        # BrainAgeNeXt parameters (using same names as config)
        zoom_min: float = 0.95,
        zoom_max: float = 1.00,
        rotate_range_x: float = 0.1,
        rotate_range_y: float = 0.1,
        rotate_range_z: float = 0.1,
        scaling_range_tio: tuple = (0.05, 0.05, 0.05),
        rotation_range_tio: tuple = (5, 5, 5),
        translation_range: float = 10.0,
        bias_field_range: tuple[float, float] = (-0.5, 0.1),
        bias_field_degree: int = 5,
        motion_degrees: float = 3,
        motion_translation: float = 5,
        motion_num_transforms: int = 4,
        ghost_num: tuple[int, int] = (1, 4),
        ghost_intensity: tuple[float, float] = (0.1, 0.6),
        max_res_aniso: float = 8.0,
        atlas_res: float = 1.0,
        thickness: float | None = None,
        # label config
        generation_labels: np.ndarray | None = None,
        n_neutral_labels: int | None = None,
        output_labels: np.ndarray | None = None,
        # tumor generation
        tumor_perlin_res: list[int] = [4, 4, 4],
        tumor_percentile_range: tuple[float, float] = (90.0, 99.6),
        tumor_size_factor_range: tuple[float, float] = (0.5, 2.0),
        tumor_use_fluid_dynamics: bool = True,
        # toggles
        use_sample: bool = True,
        use_hemisphere_aware_flip: bool = True,
        use_dynamic_resolution: bool = True,
        use_intensity_clip_normalize: bool = True,
        use_torchio_transforms: bool = True,
        use_znormalization: bool = True,
        n_channels: int = 1,
        use_specific_stats_for_channel: bool = False,
        output_shape: Sequence[int] | int | None = None,
        use_random_cropping: bool = False,
        return_gradients: bool = False,
        return_segmentation: bool = False,
        device: torch.device | str | None = None,
    ):
        # trivial fields
        self.image_key, self.label_key = "image", "labels"
        self.prob = prob
        self.prior_means, self.prior_stds = prior_means, prior_stds
        self.distribution = distribution

        # spatial ranges
        self.rotate_rad = np.deg2rad(rotation_range)
        self.scale_bounds = scaling_range
        self.shear_bounds = shear_bounds
        self.translation_bounds = translation_bounds

        # BrainAgeNeXt spatial parameters (using same names as config)
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max
        self.rotate_range_x = rotate_range_x
        self.rotate_range_y = rotate_range_y
        self.rotate_range_z = rotate_range_z
        self.scaling_range_tio = scaling_range_tio
        self.rotation_range_tio = rotation_range_tio
        self.translation_range = translation_range

        # intensity ranges
        self.contrast_range = contrast_range
        self.log_gamma_std = log_gamma_std
        self.shift_offset = shift_offset
        self.hist_control_points = hist_control_points

        # BrainAgeNeXt intensity parameters (using same names as config)
        self.bias_field_range = bias_field_range
        self.bias_field_degree = bias_field_degree

        # artefacts
        self.noise_mean, self.noise_std = noise_mean, noise_std
        self.rician_std, self.gibbs_alpha = rician_std, gibbs_alpha
        self.blur_sigma = blur_sigma
        self.bias_field_rng = bias_field_rng

        # BrainAgeNeXt artifact parameters (using same names as config)
        self.motion_degrees = motion_degrees
        self.motion_translation = motion_translation
        self.motion_num_transforms = motion_num_transforms
        self.ghost_num = ghost_num
        self.ghost_intensity = ghost_intensity

        # resolution
        self.min_res, self.max_res_iso, self.max_res_aniso = min_res, max_res_iso, max_res_aniso
        self.atlas_res = atlas_res
        self.thickness = thickness or atlas_res

        # label arrays (use explicit None test – avoids NumPy truth-value error)
        self.generation_labels = generation_labels if generation_labels is not None else GENERATION_LABELS
        self.n_neutral_labels = n_neutral_labels if n_neutral_labels is not None else N_NEUTRAL_LABELS
        self.output_labels = output_labels if output_labels is not None else self.generation_labels

        # toggles
        self.use_sample = use_sample
        self.use_hemisphere_aware_flip = use_hemisphere_aware_flip
        self.use_dynamic_resolution = use_dynamic_resolution
        self.use_intensity_clip_normalize = use_intensity_clip_normalize
        self.use_torchio_transforms = use_torchio_transforms
        self.use_znormalization = use_znormalization
        self.n_channels = n_channels
        self.use_specific_stats_for_channel = use_specific_stats_for_channel
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.use_random_cropping = use_random_cropping
        self.return_gradients = return_gradients
        self.return_segmentation = return_segmentation

        # tumor generation parameters (using same priors as brain tissues)
        self.tumor_perlin_res = tumor_perlin_res
        self.tumor_percentile_range = tumor_percentile_range
        self.tumor_size_factor_range = tumor_size_factor_range
        self.tumor_use_fluid_dynamics = tumor_use_fluid_dynamics
        
        # pipeline runs on CPU; send batch to GPU later in training loop
        self.device = torch.device(device) if device else torch.device("cpu")
        self._build_pipeline()

    # -------------------------------------------------------------- #
    def _build_pipeline(self):
        tx = []

        # 0) optional crop/pad
        if self.use_random_cropping and self.output_shape is not None:
            tx += [
                SpatialPadd(keys=[self.image_key], spatial_size=self.output_shape,
                            mode="constant", constant_values=0),
                RandSpatialCropd(keys=[self.image_key],
                                 roi_size=self.output_shape, random_size=False),
            ]

        # 1) SPATIAL TRANSFORMS (BEFORE GMM SAMPLING)
        # Original hemisphere-aware flip
        tx.append(
            HemisphereAwareFlipD(
                keys=[self.image_key],
                generation_labels=self.generation_labels,
                n_neutral_labels=self.n_neutral_labels,
                spatial_axis=0,
                prob=self.prob["flip"],
            )
        )

        # Original affine transform
        tx.append(
            RandAffined(keys=[self.image_key], prob=self.prob["affine"],
                        rotate_range=(self.rotate_rad,)*3,
                        scale_range=(self.scale_bounds,)*3,
                        shear_range=(self.shear_bounds,)*3,
                        translate_range=(self.translation_bounds,)*3,
                        mode="nearest", padding_mode="constant")
        )

        # BrainAgeNeXt spatial transforms (before GMM sampling)
        # RandZoomd from MONAI
        if self.prob.get("zoom", 0.0) > 0.0:
            tx.append(
                RandZoomd(keys=[self.image_key], 
                         min_zoom=self.zoom_min, 
                         max_zoom=self.zoom_max, 
                         prob=self.prob["zoom"])
            )
        
        # RandRotated from MONAI  
        if self.prob.get("rotate", 0.0) > 0.0:
            tx.append(
                RandRotated(keys=[self.image_key], 
                           range_x=self.rotate_range_x, 
                           range_y=self.rotate_range_y, 
                           range_z=self.rotate_range_z, 
                           prob=self.prob["rotate"])
            )

        # keep spatially-aligned copy for segmentation GT and intensity generation
        tx.append(CopyItemsd(keys=[self.image_key], times=2, names=["seg_gt", "seg_for_intensity"]))

        # Convert seg_for_intensity to generation classes for intensity generation
        tx.append(
            ConvertLabelsD(keys=["seg_for_intensity"],
                   generation_labels=self.generation_labels,
                   output_labels=GENERATION_CLASSES,
                   background_label=0)
        )

        # Convert seg_gt to contiguous indices for the segmentation task
        output_seg_labels = np.arange(len(self.generation_labels), dtype=np.int16)
        tx.append(
            ConvertLabelsD(keys=["seg_gt"],
                   generation_labels=self.generation_labels,
                   output_labels=output_seg_labels,
                   background_label=0)
        )

        # 2) CONDITIONAL GMM SAMPLING (label → intensities)
        if self.use_sample:
            tx.append(
                SampleConditionalGMMd(
                    seg_key="seg_for_intensity", out_key=self.image_key,
                    prior_means=self.prior_means, prior_stds=self.prior_stds,
                    distribution=self.distribution)
            )

        # 2b) tumor generation (after brain tissue generation)
        if self.prob.get("tumor", 0.0) > 0.0:
            tx.append(
                RandTumorSampleConditionalGMMd(
                    seg_key="seg_for_intensity", 
                    image_key=self.image_key,
                    prior_means=self.prior_means,
                    prior_stds=self.prior_stds,
                    distribution=self.distribution,
                    prob=self.prob["tumor"],
                    perlin_res=self.tumor_perlin_res,
                    mask_percentile_min=self.tumor_percentile_range[0],
                    mask_percentile_max=self.tumor_percentile_range[1],
                    tumor_size_factor_range=self.tumor_size_factor_range,
                    use_fluid_dynamics=self.tumor_use_fluid_dynamics,
                    device=self.device,
                )
            )

        # 3) INTENSITY + ARTIFACT AUGMENTS (AFTER GMM SAMPLING)
        # BrainAgeNeXt TorchIO transforms (after GMM sampling)
        if self.use_torchio_transforms:
            torchio_transforms = []
            
            # RandomGamma from TorchIO
            if self.prob.get("gamma", 0.0) > 0.0:
                torchio_transforms.append(
                    tio.RandomGamma(log_gamma=self.log_gamma_std, p=self.prob["gamma"])
                )
            
            # ZNormalization from TorchIO (if enabled)
            if self.use_znormalization:
                torchio_transforms.append(
                    tio.ZNormalization(masking_method=lambda x: x > 0)
                )
            
            # RandomMotion from TorchIO
            if self.prob.get("motion", 0.0) > 0.0:
                torchio_transforms.append(
                    tio.RandomMotion(
                        degrees=self.motion_degrees,
                        translation=self.motion_translation,
                        num_transforms=self.motion_num_transforms,
                        p=self.prob["motion"]
                    )
                )
            
            # RandomGhosting from TorchIO
            if self.prob.get("ghost", 0.0) > 0.0:
                torchio_transforms.append(
                    tio.RandomGhosting(
                        num_ghosts=self.ghost_num,
                        intensity=self.ghost_intensity,
                        p=self.prob["ghost"]
                    )
                )
            
            # RandomNoise from TorchIO
            if self.prob.get("noise", 0.0) > 0.0:
                torchio_transforms.append(
                    tio.RandomNoise(std=(0, self.noise_std), p=self.prob["noise"])
                )
            
            # RandomSwap from TorchIO
            if self.prob.get("swap", 0.0) > 0.0:
                torchio_transforms.append(
                    tio.RandomSwap(p=self.prob["swap"])
                )
            
            # Add TorchIO transforms if any
            if torchio_transforms:
                tx.append(TorchIOTransformd(keys=[self.image_key], transforms=torchio_transforms))
        
        # BrainAgeNeXt MONAI transforms (after GMM sampling)
        # RandAdjustContrastd from MONAI (using config values)
        if self.prob.get("contrast", 0.0) > 0.0:
            tx.append(
                RandAdjustContrastd(keys=[self.image_key], 
                                   prob=self.prob["contrast"],
                                   gamma=self.contrast_range)
            )
        
        # RandBiasFieldd from MONAI (using config values)
        if self.prob.get("bias", 0.0) > 0.0:
            tx.append(
                RandBiasFieldd(keys=[self.image_key], 
                              prob=self.prob["bias"],
                              degree=self.bias_field_degree,
                              coeff_range=self.bias_field_range)
            )

        # Original intensity + artefact augments (keep existing ones that aren't replaced)
        tx += [
            RandGammaD(keys=[self.image_key], log_gamma_std=self.log_gamma_std,
                       prob=self.prob.get("gamma_original", 0.0)),  # Renamed to avoid conflict
            RandScaleIntensityd(keys=[self.image_key], prob=self.prob.get("scale_int", 0.0),
                                factors=self.contrast_range),
            RandShiftIntensityd(keys=[self.image_key], prob=self.prob.get("shift_int", 0.0),
                                offsets=self.shift_offset),
            RandHistogramShiftd(keys=[self.image_key], prob=self.prob.get("hist_shift", 0.0),
                                num_control_points=self.hist_control_points),
            RandGaussianNoised(keys=[self.image_key], prob=self.prob.get("noise_gaussian", 0.0),
                               mean=self.noise_mean, std=self.noise_std),
            RandRicianNoised(keys=[self.image_key], prob=self.prob.get("rician", 0.0),
                             std=self.rician_std),
            RandGibbsNoised(keys=[self.image_key], prob=self.prob.get("gibbs", 0.0),
                            alpha=self.gibbs_alpha),
            RandGaussianSmoothd(keys=[self.image_key], prob=self.prob.get("blur", 0.0),
                                sigma_x=(0.0, self.blur_sigma),
                                sigma_y=(0.0, self.blur_sigma),
                                sigma_z=(0.0, self.blur_sigma)),
            RandBiasFieldd(keys=[self.image_key], prob=self.prob.get("bias_original", 0.0),
                           coeff_range=self.bias_field_rng),
        ]

        # 4) resolution simulation
        tx.append(
            DynamicResolutionD(keys=[self.image_key],
                                atlas_res=self.atlas_res,
                                max_res_iso=self.max_res_iso,
                                max_res_aniso=self.max_res_aniso,
                                thickness_factor=self.thickness,
                                randomise_res=True,
                                prob=self.prob.get("resolution", 0.0))
        )

        # 5) optional clip / normalise
        if self.use_intensity_clip_normalize:
            tx.append(
                IntensityClipNormalizeD(keys=[self.image_key],
                                        clip_percentiles=(1.0, 99.0),
                                        normalise=True, gamma_std=0.2,
                                        separate_channels=True, prob=0.95)
            )

        # 6) zero background (after all augmentations)
        tx.append(ZeroBackgroundd(img_key=self.image_key, seg_key="seg_gt"))

        # 7) clean-up & tensor-convert
        keys_to_delete = ["seg_for_intensity"]  # Always remove the temporary intensity segmentation
        if not self.return_segmentation:
            keys_to_delete.append("seg_gt")
        
        if keys_to_delete:
            tx.append(DeleteItemsd(keys=keys_to_delete))
            
        tensor_keys = [self.image_key]
        if self.return_segmentation:
            tensor_keys.append("seg_gt")

        tx.append(ToTensord(keys=tensor_keys, allow_missing_keys=True))

        self.transform = Compose(tx)

    # -------------------------------------------------------------- #
    def __call__(self, sample: dict) -> dict:
        # save original labels if caller asked for remap later
        if not np.array_equal(self.generation_labels, self.output_labels):
            orig = sample[self.image_key]
            sample[self.image_key + "_original_labels"] = (
                orig.clone() if torch.is_tensor(orig) else orig.copy()
            )

        out = self.transform(sample)

        if not np.array_equal(self.generation_labels, self.output_labels):
            out[self.label_key] = out.pop(self.image_key + "_original_labels")

        return out