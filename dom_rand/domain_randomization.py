"""
GPU-aware domain-randomisation pipeline for 3-D brain MR images.

• Combines MONAI (fast, GPU-ready) and TorchIO (advanced MRI artefacts).
• Probabilities of every transform are configurable.
• Can be instantiated once and safely reused by multiple workers.

Typical use
-----------
    dr = DomainRandomizer(device=torch.device("cuda"), **cfg.get("domain_randomization", {}))
    train_ds = BADataset(..., transform=dr, mode="train")
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Any

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
    RandSpatialCropd,
    ToTensord,
)
from brain_age_pred.dom_rand.custom_transformations import (
    RandomResolutionD,
    RandGammaD,
)


class DomainRandomizer:
    """
    Compose random geometric + intensity + artefact transforms.

    Parameters
    ----------
    device          : torch.device – tensors & kernel execution device.
    image_key       : str          – dictionary key that stores the image.
    use_torchio     : bool         – enable slow, heavy TorchIO artefacts.
    transform_probs : Dict[str,float] – override default probabilities.
    current_epoch / max_epochs
                    : int          – updated externally for curriculum if desired.
    Other kwargs correspond to bound ranges for the transforms.
    """

    # ---------------------- default hyper-parameters ---------------------- #
    _DEFAULT_PROBS: Dict[str, float] = {
        "flip": 0.5,
        "affine": 0.8,
        "elastic": 0.4,
        "contrast": 0.6,
        "gamma": 0.5,
        "blur": 0.4,
        "bias": 0.5,
        "resolution": 0.5,
        "spike": 0.2,
        "ghost": 0.3,
        "crop": 1.0,
    }

    def __init__(
        self,
        *,
        device: torch.device,
        image_key: str = "image",
        ### probabilities ###################################################
        transform_probs: Optional[Dict[str, float]] = None,
        ### geometric #######################################################
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: float = 10.0,       # deg
        shearing_bounds: float = 0.05,
        ### intensity #######################################################
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        log_gamma_std: float = 0.2,
        bias_field_range: Tuple[float, float] = (0.0, 0.4),
        ### resolution / artefacts ##########################################
        max_res_iso: float = 3.0,
        ### misc ############################################################
        output_shape: Tuple[int, int, int] | None = (150, 200, 150),
        use_torchio: bool = False,
        current_epoch: int = 0,
        max_epochs: int = 200,
        **unused,
    ):
        self.image_key     = image_key
        self.device        = device
        self.current_epoch = current_epoch
        self.max_epochs    = max_epochs
        self.use_tio       = use_torchio

        # merge probabilities
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
        self.output_shape   = output_shape

        # build internal pipelines
        self._build_monai_pipeline()
        self._build_torchio_pipeline()

    # --------------------------- build pipelines --------------------------- #
    def _build_monai_pipeline(self) -> None:
        """Fast GPU-ready transforms via MONAI."""
        deg2rad = np.pi / 180
        tfms = []

        # 1. random LR flip
        tfms.append(RandFlipd(
            keys=[self.image_key], prob=self.prob["flip"], spatial_axis=0
        ))

        # 2. affine (rotate, scale, shear, translate)
        tfms.append(RandAffined(
            keys=[self.image_key],
            prob=self.prob["affine"],
            rotate_range=(deg2rad * self.rotation_range) * 3,
            scale_range=(self.scaling_range[1] - 1) * 3,
            shear_range=(self.shearing_bounds) * 3,
            mode="bilinear",
        ))

        # 3. contrast adjust
        tfms.append(RandAdjustContrastd(
            keys=[self.image_key],
            prob=self.prob["contrast"],
            gamma=self.contrast_range,
        ))

        # 4. gamma exponent
        tfms.append(RandGammaD(
            keys=[self.image_key],
            log_gamma_std=self.log_gamma_std,
            prob=self.prob["gamma"],
        ))

        # 5. gaussian blur
        tfms.append(RandGaussianSmoothd(
            keys=[self.image_key],
            prob=self.prob["blur"],
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ))

        # 6. bias field
        tfms.append(RandBiasFieldd(
            keys=[self.image_key],
            prob=self.prob["bias"],
            coeff_range=self.bias_field_rng,
        ))

        # 7. simulate lower resolution
        tfms.append(RandomResolutionD(
            keys=[self.image_key],
            min_res=1.0,
            max_res_iso=self.max_res_iso,
            prob=self.prob["resolution"],
        ))

        # 8. (optional) crop to fixed ROI
        if self.output_shape is not None:
            tfms.append(RandSpatialCropd(
                keys=[self.image_key],
                roi_size=self.output_shape,
                random_center=True,
                random_size=False,
            ))

        # 9. convert to tensor
        tfms.append(ToTensord(keys=[self.image_key]))

        self.monai = Compose(tfms)
        # push MONAI transforms that support it to GPU
        if self.device.type == "cuda":
            for t in self.monai.transforms:
                if hasattr(t, "set_device"):
                    t.set_device(self.device)

    def _build_torchio_pipeline(self) -> None:
        """Optional heavy-weight MRI artefacts (elastic, spike, ghosting)."""
        if not self.use_tio:
            self.tio = None
            return

        tfms: list[tio.Transform] = [
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=5.0,
                locked_borders=2,
                p=self.prob["elastic"],
            ),
            tio.RandomSpike(
                num_spikes=(1, 6), intensity=(0.1, 0.6), p=self.prob["spike"]
            ),
            tio.RandomGhosting(
                num_ghosts=(2, 10), axes=(0, 1, 2), p=self.prob["ghost"]
            ),
        ]
        self.tio = tio.Compose(tfms)

    # --------------------------------------------------------------------- #
    #                              call                                     #
    # --------------------------------------------------------------------- #
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = sample[self.image_key]

        # TorchIO transforms (CPU) – elastic, spike, ghosting
        if self.tio is not None:
            subj = tio.Subject({self.image_key: tio.ScalarImage(tensor=img)})
            img = self.tio(subj)[self.image_key].data

        # MONAI transforms (GPU capable)
        img = self.monai({self.image_key: img})[self.image_key]

        # Keep image on the same device as where it was processed
        sample[self.image_key] = img

        # Don't move tensors - let the trainer handle device placement
        # Only ensure other fields are on the same device as the image
        for k in ("age", "weight"):
            if k in sample and sample[k].device != img.device:
                sample[k] = sample[k].to(img.device)

        return sample