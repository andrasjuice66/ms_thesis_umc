import torchio
from monai.transforms import LoadImaged, EnsureSizeOfHdr, ReshapeImage, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd
import torch
import torch.nn as nn
from monai.transforms import Compose



monai_transforms = [
    LoadImaged(keys=["img"], ensure_channel_first=True),
    EnsureSizeOfHdr(keys=["img"]),
    ReshapeImage(keys=["img"]),
    Spacingd(keys=["img"], pixdim=(p, p, p)),
    CropForegroundd(keys=["img"], allow_smaller=True, source_key="img"),
    SpatialPadd(keys=["img"], spatial_size=(x, y, z)),
    CenterSpatialCropd(keys=["img"], roi_size=(x, y, z)),]

train_torchio_transforms = torchio.transforms.Compose(
    [
        torchio.transforms.RandomGamma(log_gamma=0.8, keys=["img"], include=['img'], p=0.5),
        torchio.transforms.ZNormalization(masking_method=lambda x: x > 0, keys=["img"], include=['img']),
        torchio.transforms.RandomAffine(scales=(0.05,0.05,0.05), degrees=(5,5,5), translation=(10,10,10), include=['img'], p=0.5),
        torchio.transforms.RandomMotion(degrees=(3), translation=(5), num_transforms=4, keys=["img"], p=0.5, include=['img']),
        torchio.transforms.RandomGhosting(num_ghosts=(1, 4), intensity=(0.1, 0.6), keys=["img"], p=0.5, include=['img']),
        torchio.RandomNoise(keys=["img"], std=[0,0.5], p=0.5, include=['img']),
        torchio.RandomSwap(keys=["img"], p=0.5, include=['img']),
    ]
)

val_torchio_transforms = torchio.transforms.Compose(
    [
        torchio.transforms.ZNormalization(masking_method=lambda x: x > 0, keys=["img"],include=['img']),
    ]
)

train_transforms = Compose(monai_transforms + [train_torchio_transforms] + monai_transforms_random)
val_transforms = Compose(monai_transforms + [val_torchio_transforms])

optimizer = torch.optim.Adam(model.parameters(), 0.5e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
loss_function = nn.HuberLoss(delta=5.0)
