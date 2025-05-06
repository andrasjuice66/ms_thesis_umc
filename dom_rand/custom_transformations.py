import torch
import random
import torch.nn.functional as F
from monai.transforms.transform import MapTransform
from scipy.ndimage import gaussian_filter
import numpy as np



class RandGammaD(MapTransform):
    """
    Voxel‐wise gamma exponentiation: gamma = exp(N(0, log_gamma_std^2))
    """
    def __init__(self, keys, log_gamma_std: float = 0.2, prob: float = 0.5):
        super().__init__(keys)
        self.log_gamma_std = log_gamma_std
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]
        if random.random() < self.prob:
            # sample exponent in log‐domain
            log_g = torch.randn(1, device=img.device) * self.log_gamma_std
            gamma = torch.exp(log_g).item()
            d[self.keys[0]] = img.pow(gamma)
        return d


class RandomResolutionD(MapTransform):
    """
    Simulate acquisition at random low‐resolution and resample back to original.
    """
    def __init__(self,
                 keys,
                 min_res: float = 1.0,
                 max_res_iso: float = 4.0,
                 prob: float = 0.5):
        super().__init__(keys)
        self.min_res = min_res
        self.max_res_iso = max_res_iso
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]  # shape = (C,H,W,D)
        if random.random() < self.prob:
            # assume original voxel size = 1.0
            shape = img.shape[1:]
            # pick a random isotropic low‐res spacing
            lr = random.uniform(self.min_res, self.max_res_iso)
            # approximate slice‐thickness blur using scipy
            img_np = img.cpu().numpy()
            # apply gaussian_filter channel-wise
            blurred = np.stack([
                gaussian_filter(img_np[c], sigma=lr)
                for c in range(img_np.shape[0])
            ], axis=0)
            img = torch.from_numpy(blurred).to(img.device).type(img.dtype)
            # downsample to low resolution
            new_size = [max(1, int(s / lr)) for s in shape]
            img = F.interpolate(
                img.unsqueeze(0),
                size=new_size,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
            # upsample back to original grid
            img = F.interpolate(
                img.unsqueeze(0),
                size=shape,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
            d[self.keys[0]] = img
        return d
