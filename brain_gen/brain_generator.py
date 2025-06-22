# brain_age_pred/dataset/synthseg_generator.py

import numpy as np
from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandRicianNoised,
    RandGibbsNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandHistogramShiftd,
    ToTensord,
    RandSpatialCropd,
    SpatialPadd,
)
from brain_age_pred.dataset.custom_transformations import (
    RandomResolutionD, 
    RandGammaD,
    HemisphereAwareFlipD,
    DynamicResolutionD, 
    IntensityClipNormalizeD,
    ConvertLabelsD, 
    ImageGradientsD
)
from brain_age_pred.brain_gen.gen_image_from_labels import SampleConditionalGMMd, MultiChannelSampleConditionalGMMd

# Default SynthSeg label configuration
DEFAULT_GENERATION_LABELS = np.array([
    0,   # background
    14, 15, 16, 24, 77, 85,  # neutral structures (ventricles, brainstem, etc.)
    2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28,  # left structures
    41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60  # right structures
])
DEFAULT_N_NEUTRAL_LABELS = 7  # background + 6 neutral structures


class BABrainGenerator:
    """
    Turn a segmentation map into a *new*, domain‐randomized synthetic brain volume.
    Enhanced with SynthSeg-style hemisphere-aware flipping, dynamic resolution, and intensity normalization.
    Now includes: translation, cropping, label mapping, multi-channel support, and gradient computation.
    """

    def __init__(
        self,
        prior_means:  np.ndarray,
        prior_stds:   np.ndarray,
        distribution: str,
        prob:         dict,
        rotation_range: float,
        scaling_range:  float,
        shearing_bounds: float,
        # NEW: Translation bounds
        translation_bounds: float,
        contrast_range:  tuple,
        log_gamma_std:   float,
        shift_offset:    float,
        hist_control_points: int,
        noise_mean: float,
        noise_std:  float,
        rician_std: float,
        gibbs_alpha: float,
        blur_sigma:  float,
        bias_field_rng: tuple,
        min_res:       float,
        max_res_iso:   float,
        # SynthSeg-style parameters
        generation_labels: np.ndarray = None,
        n_neutral_labels: int = None,
        # NEW: Output labels for remapping
        output_labels: np.ndarray = None,
        use_hemisphere_aware_flip: bool = True,
        use_dynamic_resolution: bool = True,
        use_intensity_clip_normalize: bool = True,
        max_res_aniso: float = 8.0,
        atlas_res: float = 1.0,
        intensity_clip_value: float = 300.0,
        intensity_gamma_std: float = 0.5,
        # NEW: Multi-channel support
        n_channels: int = 1,
        use_specific_stats_for_channel: bool = False,
        # NEW: Cropping parameters
        output_shape: tuple = None,
        use_random_cropping: bool = False,
        # NEW: Gradient computation
        return_gradients: bool = False,
        # NEW: Slice thickness simulation
        thickness: float = None,
    ):
        self.image_key    = "image"
        self.label_key    = "labels"  # For output labels
        self.prior_means  = prior_means
        self.prior_stds   = prior_stds
        self.distribution = distribution
        self.prob         = prob
        
        # Spatial transform parameters
        self.rotate_rad   = np.deg2rad(rotation_range)
        self.scaling_rng  = scaling_range
        self.shearing    = shearing_bounds
        self.translation_bounds = translation_bounds  # NEW
        
        # Intensity transform parameters
        self.contrast_rng= contrast_range
        self.log_gamma_std = log_gamma_std
        self.shift_offset  = shift_offset
        self.hist_control_points = hist_control_points
        
        # Noise parameters
        self.noise_mean  = noise_mean
        self.noise_std   = noise_std
        self.rician_std  = rician_std
        self.gibbs_alpha = gibbs_alpha
        self.blur_sigma  = blur_sigma
        self.bias_field_rng = bias_field_rng
        
        # Resolution parameters
        self.min_res     = min_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        self.atlas_res = atlas_res
        self.thickness = thickness if thickness is not None else atlas_res  # NEW
        
        # SynthSeg-style enhancements
        self.generation_labels = generation_labels if generation_labels is not None else DEFAULT_GENERATION_LABELS
        self.n_neutral_labels = n_neutral_labels if n_neutral_labels is not None else DEFAULT_N_NEUTRAL_LABELS
        self.output_labels = output_labels if output_labels is not None else self.generation_labels  # NEW
        self.use_hemisphere_aware_flip = use_hemisphere_aware_flip
        self.use_dynamic_resolution = use_dynamic_resolution
        self.use_intensity_clip_normalize = use_intensity_clip_normalize
        self.intensity_clip_value = intensity_clip_value
        self.intensity_gamma_std = intensity_gamma_std
        
        # NEW: Multi-channel support
        self.n_channels = n_channels
        self.use_specific_stats_for_channel = use_specific_stats_for_channel
        
        # NEW: Cropping parameters
        self.output_shape = output_shape
        self.use_random_cropping = use_random_cropping
        
        # NEW: Gradient computation
        self.return_gradients = return_gradients

        self._build_pipeline()

    def _build_pipeline(self):
        tx = []

        # 0) OPTIONAL: Random cropping of input labels (SynthSeg-style)
        if self.use_random_cropping and self.output_shape is not None:
            # First pad to ensure we have enough space for cropping
            tx.append(
                SpatialPadd(
                    keys=[self.image_key],
                    spatial_size=self.output_shape,
                    mode="constant",
                    constant_values=0,
                )
            )
            # Then randomly crop to desired output shape
            tx.append(
                RandSpatialCropd(
                    keys=[self.image_key],
                    roi_size=self.output_shape,
                    random_size=False,
                )
            )

        # 1) Spatial transforms on the *segmentation*
        if self.use_hemisphere_aware_flip:
            # Use hemisphere-aware flipping instead of simple random flip
            tx.append(
                HemisphereAwareFlipD(
                    keys=[self.image_key],
                    generation_labels=self.generation_labels,
                    n_neutral_labels=self.n_neutral_labels,
                    spatial_axis=0,  # Left-right axis
                    prob=self.prob["flip"],
                )
            )
        else:
            # Use standard MONAI flip
            tx.append(
                RandFlipd(
                    keys=[self.image_key],
                    prob=self.prob["flip"],
                    spatial_axis=(0, 1, 2),
                )
            )
        
        # Affine transforms - NOW WITH TRANSLATION!
        tx.append(
            RandAffined(
                keys=[self.image_key],
                prob=self.prob["affine"],
                rotate_range=(self.rotate_rad,) * 3,
                scale_range=(self.scaling_rng - 1,) * 3,
                shear_range=(self.shearing,) * 3,
                translate_range=(self.translation_bounds,) * 3,  # NEW: Translation!
                mode="nearest",   # preserve labels
                padding_mode="constant",
                constant_values=0,
            )
        )

        # 2) Sample intensities from labels → synthetic image
        if self.n_channels == 1:
            # Single channel - standard approach
            tx.append(
                SampleConditionalGMMd(
                    seg_key=self.image_key,
                    out_key=self.image_key,
                    prior_means=self.prior_means,
                    prior_stds =self.prior_stds,
                    distribution=self.distribution,
                )
            )
        else:
            # Multi-channel generation
            tx.append(
                MultiChannelSampleConditionalGMMd(  # NEW: Multi-channel version
                    seg_key=self.image_key,
                    out_key=self.image_key,
                    prior_means=self.prior_means,
                    prior_stds=self.prior_stds,
                    distribution=self.distribution,
                    n_channels=self.n_channels,
                    use_specific_stats_for_channel=self.use_specific_stats_for_channel,
                )
            )

        # 3) Intensity clipping and normalization (SynthSeg-style)
        if self.use_intensity_clip_normalize:
            tx.append(
                IntensityClipNormalizeD(
                    keys=[self.image_key],
                    clip_value=self.intensity_clip_value,
                    normalise=True,
                    gamma_std=self.intensity_gamma_std,
                    separate_channels=True,
                    prob=0.95,
                )
            )

        # 4) Additional intensity‐space augmentations
        tx.extend([
            RandAdjustContrastd(
                keys=[self.image_key],
                prob=self.prob["contrast"],
                gamma=self.contrast_rng,
            ),
            RandGammaD(
                keys=[self.image_key],
                log_gamma_std=self.log_gamma_std,
                prob=self.prob["gamma"],
            ),
            RandScaleIntensityd(
                keys=[self.image_key],
                prob=self.prob["scale_int"],
                factors=self.contrast_rng,
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

        # 5) Add artifacts & simulate resolution
        tx.extend([
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
        ])
        
        # 6) Resolution simulation - WITH SLICE THICKNESS
        if self.use_dynamic_resolution:
            # Dynamic resolution sampling (SynthSeg-style)
            tx.append(
                DynamicResolutionD(
                    keys=[self.image_key],
                    atlas_res=self.atlas_res,
                    max_res_iso=self.max_res_iso,
                    max_res_aniso=self.max_res_aniso,
                    thickness=self.thickness,  # NEW: Slice thickness
                    randomise_res=True,
                    prob=self.prob["resolution"],
                )
            )
        else:
            # Use the original static resolution transform
            tx.append(
                RandomResolutionD(
                    keys=[self.image_key],
                    min_res=self.min_res,
                    max_res_iso=self.max_res_iso,
                    prob=self.prob["resolution"],
                )
            )

        # 7) NEW: Gradient computation (optional)
        if self.return_gradients:
            tx.append(
                ImageGradientsD(
                    keys=[self.image_key],
                    method="sobel",
                    normalize=True,
                )
            )

        # 8) NEW: Label remapping (convert generation labels to output labels)
        if not np.array_equal(self.generation_labels, self.output_labels):
            tx.append(
                ConvertLabelsD(
                    keys=[self.image_key + "_original_labels"],  # We'll need to preserve original labels
                    generation_labels=self.generation_labels,
                    output_labels=self.output_labels,
                    background_label=0,
                )
            )

        # 9) Convert to tensor
        tx.append(ToTensord(keys=[self.image_key]))

        self.transform = Compose(tx)

    def __call__(self, sample: dict) -> dict:
        # Store original labels if we need label remapping
        if not np.array_equal(self.generation_labels, self.output_labels):
            sample[self.image_key + "_original_labels"] = sample[self.image_key].copy()
        
        result = self.transform(sample)
        
        # Return both image and labels if label conversion was applied
        if not np.array_equal(self.generation_labels, self.output_labels):
            result[self.label_key] = result.pop(self.image_key + "_original_labels")
        
        return result