"""
Domain–randomised brain-image generator (SynthSeg-style)
-------------------------------------------------------
Takes a *segmentation* as input and returns a synthetic MRI volume that
shares the same anatomy but random contrast, artefacts, resolution, etc.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Mapping, Hashable, Dict

import torch
from monai.transforms import (
    Compose, RandFlipd, RandAffined, RandAdjustContrastd, RandBiasFieldd,
    RandGaussianSmoothd, RandGaussianNoised, RandRicianNoised, RandGibbsNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandHistogramShiftd, ToTensord,
    RandSpatialCropd, SpatialPadd,  CopyItemsd
)


# project imports -------------------------------------------------------
from brain_age_pred.dataset.custom_transformations import (
    RandomResolutionD, RandGammaD, HemisphereAwareFlipD, DynamicResolutionD,
    IntensityClipNormalizeD, ConvertLabelsD,
)
from brain_age_pred.brain_gen.gen_image_from_labels import (
    SampleConditionalGMMd, MultiChannelSampleConditionalGMMd, 
)
from brain_age_pred.brain_gen.labels import (
    GENERATION_LABELS, GENERATION_CLASSES, N_NEUTRAL_LABELS,
)
# ---------------------------------------------------------------
# After your existing imports – add these two one-liners
# ---------------------------------------------------------------
from monai.transforms import Lambdad, MaskIntensityd


# 1. copy class_map  →  brain_mask
# 2. convert to binary mask in-place
MakeBrainMaskd = Compose([
    CopyItemsd(keys=["class_map"], times=1, names=["brain_mask"]),
    Lambdad(keys=["brain_mask"], func=lambda x: (x > 0).float()),
])

# === set background to zero at the end ==========================
ApplyBrainMaskd = MaskIntensityd(
    keys="image",
    mask_key="brain_mask",
)


class BABrainGenerator:
    """
    Build a MONAI ``Compose`` that maps a dict
      {"image": <segmentation nparray or tensor>}
    to
      {"image": <synthetic MRI tensor> [, "labels": <original seg>]}.

    Key extras over vanilla SynthSeg:
      • hemisphere-aware flipping
      • dynamic resolution & slice-thickness
      • intensity normalisation / clipping
      • optional multi-channel output
      • optional gradient output
    """

    # ------------------------------------------------------------------ #
    # constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        prior_means:  np.ndarray,
        prior_stds:   np.ndarray,
        distribution: str,
        prob:         dict,

        # spatial augmentation
        rotation_range:   float,
        scaling_range:    float,
        shear_bounds:     float,
        translation_bounds: float,

        # intensity augmentation
        contrast_range:   tuple[float, float],
        log_gamma_std:    float,
        shift_offset:     float,
        hist_control_points: int,

        # artefacts
        noise_mean: float,
        noise_std:  float,
        rician_std: float,
        gibbs_alpha: float,
        blur_sigma:  float,
        bias_field_rng: tuple[float, float],

        # resolution
        min_res: float,
        max_res_iso: float,
        max_res_aniso: float = 8.0,
        atlas_res: float = 1.0,
        thickness: float | None = None,

        # SynthSeg label config
        generation_labels: np.ndarray | None = None,
        n_neutral_labels:  int | None = None,
        output_labels:     np.ndarray | None = None,

        # toggles
        use_hemisphere_aware_flip:  bool = True,
        use_dynamic_resolution:     bool = True,
        use_intensity_clip_normalize: bool = True,
        n_channels: int = 1,
        use_specific_stats_for_channel: bool = False,
        output_shape: Sequence[int] | int | None = None,
        use_random_cropping: bool = False,
        return_gradients: bool = False,
        device: torch.device | str | None = None,
    ):
        # store trivial fields -----------------------------------------
        self.image_key, self.label_key = "image", "labels"
        self.prob = prob
        self.prior_means, self.prior_stds = prior_means, prior_stds
        self.distribution = distribution

        # --- spatial ranges ------------------------------------------
        self.rotate_rad = np.deg2rad(rotation_range)
        self.scale_bounds = scaling_range            # SynthSeg "scaling_bounds"
        self.shear_bounds = shear_bounds
        self.translation_bounds = translation_bounds

        # --- intensity ranges ----------------------------------------
        self.contrast_range  = contrast_range
        self.log_gamma_std   = log_gamma_std
        self.shift_offset    = shift_offset
        self.hist_control_points = hist_control_points

        # --- artefact params ----------------------------------------
        self.noise_mean,  self.noise_std  = noise_mean, noise_std
        self.rician_std, self.gibbs_alpha = rician_std, gibbs_alpha
        self.blur_sigma   = blur_sigma
        self.bias_field_rng = bias_field_rng

        # --- resolution ---------------------------------------------
        self.min_res, self.max_res_iso, self.max_res_aniso = min_res, max_res_iso, max_res_aniso
        self.atlas_res = atlas_res
        self.thickness = thickness if thickness is not None else atlas_res

        # --- labels --------------------------------------------------
        self.generation_labels = (
            generation_labels if generation_labels is not None else GENERATION_LABELS
        )
        self.n_neutral_labels = (
            n_neutral_labels if n_neutral_labels is not None else N_NEUTRAL_LABELS
        )
        #  default: *no* remapping
        self.output_labels = (
            output_labels if output_labels is not None else self.generation_labels
        )

        # --- options -------------------------------------------------
        self.use_hemisphere_aware_flip   = use_hemisphere_aware_flip
        self.use_dynamic_resolution      = use_dynamic_resolution
        self.use_intensity_clip_normalize= use_intensity_clip_normalize
        self.n_channels                  = n_channels
        self.use_specific_stats_for_channel = use_specific_stats_for_channel
        self.output_shape                = tuple(output_shape) if output_shape is not None else None
        self.use_random_cropping         = use_random_cropping
        self.return_gradients            = return_gradients

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        
        # build transform pipeline
        self._build_pipeline()

    # ------------------------------------------------------------------ #
    # pipeline                                                           #
    # ------------------------------------------------------------------ #
    def _build_pipeline(self):
        tx = []

        # 0) optional cropping ----------------------------------------
        if self.use_random_cropping and self.output_shape is not None:
            tx += [
                SpatialPadd(
                    keys=[self.image_key],
                    spatial_size=self.output_shape,
                    mode="constant",
                    constant_values=0,
                ),
                RandSpatialCropd(
                    keys=[self.image_key],
                    roi_size=self.output_shape,
                    random_size=False,
                ),
            ]

        # 1) spatial transforms on *segmentation* ---------------------
        if self.use_hemisphere_aware_flip:
            tx.append(
                HemisphereAwareFlipD(
                    keys=[self.image_key],
                    generation_labels=self.generation_labels,
                    n_neutral_labels=self.n_neutral_labels,
                    spatial_axis=0,
                    prob=self.prob["flip"],
                )
            )
        else:
            tx.append(
                RandFlipd(
                    keys=[self.image_key],
                    prob=self.prob["flip"],
                    spatial_axis=(0, 1, 2),
                )
            )

        tx.append(
            RandAffined(
                keys=[self.image_key],
                prob=self.prob["affine"],
                rotate_range=(self.rotate_rad,) * 3,
                scale_range=(self.scale_bounds,) * 3,   
                shear_range=(self.shear_bounds,) * 3,
                translate_range=(self.translation_bounds,) * 3,
                mode="nearest",
                padding_mode="constant",
            )
        )

        tx.append(CopyItemsd(keys=[self.image_key], times=1, names=["class_map"]))

        tx.append(
            ConvertLabelsD(
                keys=["class_map"],
                generation_labels=self.generation_labels,
                output_labels=GENERATION_CLASSES,   # <- class IDs 0‥14
                background_label=0,
            )
        )

        # 2) label → image via conditional GMM -----------------------
        if self.n_channels == 1:
            tx.append(
                SampleConditionalGMMd(
                    seg_key="class_map",
                    out_key=self.image_key,
                    prior_means=self.prior_means,
                    prior_stds=self.prior_stds,
                    distribution=self.distribution,
                )
            )
        else:
            tx.append(
                MultiChannelSampleConditionalGMMd(
                    seg_key=self.image_key,
                    out_key=self.image_key,
                    prior_means=self.prior_means,
                    prior_stds=self.prior_stds,
                    distribution=self.distribution,
                    n_channels=self.n_channels,
                    use_specific_stats_for_channel=self.use_specific_stats_for_channel,
                )
            )

        tx.append(MakeBrainMaskd)


        # 4) intensity augmentations ---------------------------------
        tx += [
            RandAdjustContrastd(keys=[self.image_key],
                                prob=self.prob["contrast"],
                                gamma=self.contrast_range),
            RandGammaD(keys=[self.image_key],
                       log_gamma_std=self.log_gamma_std,
                       prob=self.prob["gamma"]),
            RandScaleIntensityd(keys=[self.image_key],
                                prob=self.prob["scale_int"],
                                factors=self.contrast_range),
            RandShiftIntensityd(keys=[self.image_key],
                                prob=self.prob["shift_int"],
                                offsets=self.shift_offset),
            RandHistogramShiftd(keys=[self.image_key],
                                prob=self.prob["hist_shift"],
                                num_control_points=self.hist_control_points),
        ]

        # 5) artefacts -----------------------------------------------
        tx += [
            RandGaussianNoised(keys=[self.image_key],
                               prob=self.prob["noise"],
                               mean=self.noise_mean, std=self.noise_std),
            RandRicianNoised(keys=[self.image_key],
                             prob=self.prob["rician"],
                             std=self.rician_std),
            RandGibbsNoised(keys=[self.image_key],
                            prob=self.prob["gibbs"],
                            alpha=self.gibbs_alpha),
            RandGaussianSmoothd(keys=[self.image_key],
                                prob=self.prob["blur"],
                                sigma_x=(0.0, self.blur_sigma),
                                sigma_y=(0.0, self.blur_sigma),
                                sigma_z=(0.0, self.blur_sigma)),
            RandBiasFieldd(keys=[self.image_key],
                           prob=self.prob["bias"],
                           coeff_range=self.bias_field_rng),
        ]

        # 6) resolution simulation -----------------------------------
        if self.use_dynamic_resolution:
            tx.append(
                DynamicResolutionD(
                    keys=[self.image_key],
                    atlas_res=self.atlas_res,
                    max_res_iso=self.max_res_iso,
                    max_res_aniso=self.max_res_aniso,
                    thickness_factor=self.thickness,
                    randomise_res=True,
                    prob=self.prob["resolution"],
                )
            )
        else:
            tx.append(
                RandomResolutionD(
                    keys=[self.image_key],
                    min_res=self.min_res,
                    max_res_iso=self.max_res_iso,
                    prob=self.prob["resolution"],
                )
            )

        # 3) clip + normalise ----------------------------------------
        if self.use_intensity_clip_normalize:
            tx.append(
                IntensityClipNormalizeD(
                    keys=[self.image_key],
                    clip_percentiles=(1.0, 99.0),  # 1% and 99% percentiles
                    normalise=True,
                    gamma_std=0.2,
                    separate_channels=True,
                    prob=0.95,
                )
            )

        tx.append(ApplyBrainMaskd)


        # 9) tensor conversion ---------------------------------------
        tx.append(ToTensord(keys=[self.image_key]))

        self.transform = Compose(tx)

    # ------------------------------------------------------------------ #
    # callable                                                           #
    # ------------------------------------------------------------------ #
    def __call__(self, sample: dict) -> dict:
        # preserve original labels for potential remap
        if not np.array_equal(self.generation_labels, self.output_labels):
            sample[self.image_key + "_original_labels"] = sample[self.image_key].copy()

        out = self.transform(sample)
        

        if not np.array_equal(self.generation_labels, self.output_labels):
            out[self.label_key] = out.pop(self.image_key + "_original_labels")

        return out