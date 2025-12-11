# transforms_ba.py

from typing import Optional, Tuple, Union

import torchio as tio
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
    RandZoomd,
    RandRotated,
    RandAdjustContrastd,
    RandBiasFieldd,
)

def get_base_transforms(
    image_key: str = "image",
    pixdim: Union[float, Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    crop_foreground: bool = True,
    spatial_pad: Optional[Tuple[int, int, int]] = None,
    center_crop: Optional[Tuple[int, int, int]] = None,
) -> list:
    """
    Deterministic preprocessing compatible with BADataset.
    """
    keys = [image_key]
    t = [
        EnsureTyped(keys=keys),
        Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear",)),
    ]
    if crop_foreground:
        t.append(CropForegroundd(keys=keys, source_key=image_key, allow_smaller=True))
    if spatial_pad is not None:
        t.append(SpatialPadd(keys=keys, spatial_size=spatial_pad))
    if center_crop is not None:
        t.append(CenterSpatialCropd(keys=keys, roi_size=center_crop))
    return t

def get_torchio_train_aug(
    image_key: str = "image",
    p_gamma=0.5,
    p_affine=0.5,
    p_motion=0.5,
    p_ghost=0.5,
    p_noise=0.5,
    p_swap=0.5,
):
    """
    TorchIO augmentations; uses dict keys compatible with BADataset.
    """
    return tio.Compose(
        [
            tio.RandomGamma(log_gamma=0.8, include=[image_key], p=p_gamma),
 
            tio.RandomAffine(
                scales=(0.95, 1.05),
                degrees=(5, 5, 5),
                translation=(10, 10, 10),
                include=[image_key],
                p=p_affine,
            ),
            tio.RandomMotion(
                degrees=3,
                translation=5,
                num_transforms=4,
                include=[image_key],
                p=p_motion,
            ),
            tio.RandomGhosting(
                num_ghosts=(1, 4),
                intensity=(0.1, 0.6),
                include=[image_key],
                p=p_ghost,
            ),
            tio.RandomNoise(
                std=(0.0, 0.5),
                include=[image_key],
                p=p_noise,
            ),
            tio.RandomSwap(include=[image_key], p=p_swap),
        ]
    )

def get_monai_train_aug(image_key: str = "image") -> list:
    keys = [image_key]
    return [
        RandZoomd(keys=keys, min_zoom=0.95, max_zoom=1.0, prob=0.5),
        RandRotated(keys=keys, range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
        RandAdjustContrastd(keys=keys, gamma=(0.6, 3.0), prob=0.5),
        RandBiasFieldd(keys=keys, degree=5, coeff_range=(-0.5, 0.1), prob=0.5),
    ]

def get_torchio_val_aug(image_key: str = "image"):
    return tio.Compose(
        [
            # tio.ZNormalization(
            #     masking_method=lambda x: x > 0, keys=[image_key], include=[image_key]
            # ),
        ]
    )

def get_train_transforms(
    image_key: str = "image",
    pixdim: Union[float, Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    spatial_pad: Optional[Tuple[int, int, int]] = None,
    center_crop: Optional[Tuple[int, int, int]] = None,
    crop_foreground: bool = True,
) -> Compose:
    base = get_base_transforms(
        image_key=image_key,
        pixdim=pixdim,
        crop_foreground=crop_foreground,
        spatial_pad=spatial_pad,
        center_crop=center_crop,
    )
    tio_train = get_torchio_train_aug(image_key=image_key)
    monai_rand = get_monai_train_aug(image_key=image_key)
    return Compose(base + [tio_train] + monai_rand)

def get_val_transforms(
    image_key: str = "image",
    pixdim: Union[float, Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    spatial_pad: Optional[Tuple[int, int, int]] = None,
    center_crop: Optional[Tuple[int, int, int]] = None,
    crop_foreground: bool = True,
) -> Compose:
    base = get_base_transforms(
        image_key=image_key,
        pixdim=pixdim,
        crop_foreground=crop_foreground,
        spatial_pad=spatial_pad,
        center_crop=center_crop,
    )
    tio_val = get_torchio_val_aug(image_key=image_key)
    return Compose(base + [tio_val])