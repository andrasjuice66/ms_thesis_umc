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
from typing import Dict, Tuple, Optional

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

class DomainRandomizer:
    """
    Random geometric, intensity and artefact transforms for 3-D MRI volumes.
    """

    _DEFAULT_PROBS: Dict[str, float] = {
        # geometric
        "flip"      : 0.5,
        "affine"    : 0.8,
        "elastic"   : 0.4,   # TorchIO
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
        "spike"     : 0.2,   # TorchIO
        "ghost"     : 0.3,   # TorchIO
        # misc
        "crop"      : 1.0,
    }

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        device=torch.device("cuda"), 
        image_key: str = "image",
        # probability overrides
        transform_probs: Optional[Dict[str, float]] = None,
        # Add new parameters for progressive randomization
        progressive_epochs: int = 50,  
        progressive_start: float = 0.2,  
        # geometric ranges
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: float = 10.0,          # degrees
        shearing_bounds: float = 0.05,
        # intensity
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        log_gamma_std: float = 0.2,
        bias_field_range: Tuple[float, float] = (0.0, 0.4),
        # resolution / artefacts
        max_res_iso: float = 3.0,
        coarse_dropout_size: Tuple[int, int, int] = (20, 20, 20),
        # misc
        output_shape: Tuple[int, int, int] = (182, 218, 182),
        use_torchio: bool = False,
        **unused,
    ):
        self.image_key     = image_key
        self.device        = device
        self.use_tio       = use_torchio

        # merge / override probabilities
        self.prob = {**DomainRandomizer._DEFAULT_PROBS}
        if transform_probs:
            self.prob.update(transform_probs)

        # store bounds
        self.scaling_range  = scaling_range
        self.rotation_range = rotation_range
        self.shearing_bounds = shearing_bounds
        self.contrast_range = contrast_range
        self.log_gamma_std  = log_gamma_std
        self.bias_field_rng = bias_field_range
        self.max_res_iso    = max_res_iso
        self.coarse_size    = coarse_dropout_size
        self.output_shape   = output_shape

        # Store progressive randomization parameters
        self.progressive_epochs = progressive_epochs
        self.progressive_start = progressive_start
        
        # Store original probabilities for reference
        self.original_probs = {**self.prob}
        
        # Initialize with starting probabilities
        self.current_epoch = 0
        self._update_progressive_probs()

        # build transform pipelines
        self._build_monai_pipeline()
        self._build_torchio_pipeline()

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
                spatial_axis=0,
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
                offsets=(-0.1, 0.1),
            ),
            RandHistogramShiftd(
                keys=[self.image_key],
                prob=self.prob["hist_shift"],
                num_control_points=(5, 10),
            ),
        ])

        # 3. noise / artefacts
        tfms.extend([
            RandGaussianNoised(
                keys=[self.image_key],
                prob=self.prob["noise"],
                mean=0.0,
                std=0.05,
            ),
            RandRicianNoised(
                keys=[self.image_key],
                prob=self.prob["rician"],
                std=0.05,
            ),
            RandGibbsNoised(
                keys=[self.image_key],
                prob=self.prob["gibbs"],
                alpha=(0.0, 1.0),
            ),
            RandGaussianSmoothd(
                keys=[self.image_key],
                prob=self.prob["blur"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            ),
            RandBiasFieldd(
                keys=[self.image_key],
                prob=self.prob["bias"],
                coeff_range=self.bias_field_rng,
            ),
            RandomResolutionD(
                keys=[self.image_key],
                min_res=1.0,
                max_res_iso=self.max_res_iso,
                prob=self.prob["resolution"],
            ),
            RandCoarseDropoutd(
                keys=[self.image_key],
                prob=self.prob["coarse_do"],
                holes=8,
                spatial_size=self.coarse_size,
                fill_value=0.0,
            ),
        ])

        # 4. optional crop to ROI
        if self.output_shape is not None:
            tfms.append(
                RandSpatialCropd(
                    keys=[self.image_key],
                    roi_size=self.output_shape,
                    random_center=True,
                    random_size=False,
                )
            )

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
                num_control_points=7,
                max_displacement=5.0,
                locked_borders=2,
                p=self.prob["elastic"],
            ),
            tio.RandomSpike(
                num_spikes=(1, 6),
                intensity=(0.1, 0.6),
                p=self.prob["spike"],
            ),
            tio.RandomGhosting(
                num_ghosts=(2, 10),
                axes=(0, 1, 2),
                p=self.prob["ghost"],
            ),
        ])

    def _update_progressive_probs(self) -> None:
        """Update transform probabilities based on current epoch."""
        if self.progressive_epochs <= 0:
            return
            
        # Calculate current progress (0 to 1)
        progress = min(1.0, self.current_epoch / self.progressive_epochs)
        
        # Linear interpolation between start and full probabilities
        for key in self.original_probs:
            start_prob = self.original_probs[key] * self.progressive_start
            final_prob = self.original_probs[key]
            self.prob[key] = start_prob + (final_prob - start_prob) * progress

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, epoch: int) -> None:
        """Update current epoch and adjust probabilities accordingly."""
        self._current_epoch = epoch
        self._update_progressive_probs()

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = sample[self.image_key]

        # TorchIO (CPU) artefacts
        if self.tio is not None:
            subj = tio.Subject({self.image_key: tio.ScalarImage(tensor=img)})
            img = self.tio(subj)[self.image_key].data

        # MONAI (GPU-capable) transforms
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

        # keep tensors on the same device
        sample[self.image_key] = img
        for k in ("age", "weight"):
            if k in sample and sample[k].device != img.device:
                sample[k] = sample[k].to(img.device)

        return sample