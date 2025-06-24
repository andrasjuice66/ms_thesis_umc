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


# -------------------------------------------------------------------------
# HemisphereAwareFlipD  –  version that accepts numpy *or* torch tensors
# -------------------------------------------------------------------------
from monai.transforms import MapTransform
from typing import Sequence, Mapping, Hashable, Dict
import numpy as np
import torch, random


class HemisphereAwareFlipD(MapTransform):
    """
    Random left–right flip that also swaps hemisphere‐specific labels so the
    anatomy stays consistent.

    • `generation_labels`  must list labels [neutral … left … right] in that
      exact SynthSeg order.
    • `n_neutral_labels`   = how many labels at the beginning are non-sided.
    """

    def __init__(
        self,
        keys: Sequence[str],
        generation_labels: np.ndarray,
        n_neutral_labels: int,
        spatial_axis: int = 0,   # 0 = left–right in the label volume
        prob: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.spatial_axis = spatial_axis
        self.prob = prob

        # ---------- build LUT for label swapping --------------------
        n_labels   = len(generation_labels)
        neutral    = generation_labels[:n_neutral_labels]
        n_sided    = (n_labels - n_neutral_labels) // 2
        left       = generation_labels[n_neutral_labels : n_neutral_labels + n_sided]
        right      = generation_labels[n_neutral_labels + n_sided :]

        swapped    = np.concatenate([neutral, right, left])
        max_val    = max(generation_labels.max(), swapped.max())
        lut        = np.arange(max_val + 1, dtype=np.int64)
        for a, b in zip(generation_labels, swapped):
            lut[a] = b
        self._lut = lut                    # numpy LUT, stays on CPU

    # ------------------------------------------------------------------
    def _flip_ndarray(self, arr: np.ndarray, axis: int) -> np.ndarray:
        return np.flip(arr, axis=axis).copy()     # copy: keep array C-contiguous

    def _flip_tensor(self, ten: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.flip(ten, dims=[axis])

    # ------------------------------------------------------------------
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:  # noqa: D401
        d = dict(data)

        if random.random() >= self.prob:
            return d   # no flip this time

        for key in self.key_iterator(d):
            arr = d[key]

            # --------------------------------------------------------
            # 1) flip along the requested spatial axis
            # --------------------------------------------------------
            if arr.ndim == 4:          # (C, D, H, W)
                flip_axis = self.spatial_axis + 1
            else:                      # (D, H, W)
                flip_axis = self.spatial_axis

            if isinstance(arr, torch.Tensor):
                arr_flipped = self._flip_tensor(arr, flip_axis)
            else:  # numpy
                arr_flipped = self._flip_ndarray(arr, flip_axis)

            # --------------------------------------------------------
            # 2) swap left/right labels – **only** for the first key,
            #    which is assumed to be the segmentation map.
            # --------------------------------------------------------
            if key == self.keys[0]:
                if isinstance(arr_flipped, torch.Tensor):
                    arr_int   = arr_flipped.long()
                    arr_swapped = torch.as_tensor(self._lut, device=arr_int.device)[arr_int]
                    arr_flipped = arr_swapped.type(arr_flipped.dtype)
                else:
                    arr_flipped = self._lut[arr_flipped.astype(np.int64)].astype(arr_flipped.dtype)

            d[key] = arr_flipped

        return d
class DynamicResolutionD(MapTransform):
    """
    Dynamic resolution sampling following SynthSeg's approach.
    
    This transform:
    1. Samples a random resolution per batch/sample
    2. Applies blur corresponding to the sampled resolution
    3. Optionally downsamples and upsamples to simulate acquisition
    """
    
    def __init__(self,
                 keys,
                 atlas_res: float = 1.0,
                 max_res_iso: float = 4.0,
                 max_res_aniso: float = 8.0,
                 thickness_factor: float = 1.0,
                 randomise_res: bool = True,
                 prob: float = 0.5):
        super().__init__(keys)
        self.atlas_res = atlas_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        self.thickness_factor = thickness_factor
        self.randomise_res = randomise_res
        self.prob = prob
    
    def _sample_resolution(self):
        """Sample random resolution following SynthSeg logic."""
        if not self.randomise_res:
            return self.atlas_res, self.atlas_res
        
        # Sample isotropic resolution
        if random.random() < 0.7:  # 70% chance for isotropic
            resolution = random.uniform(self.atlas_res, self.max_res_iso)
            blur_res = resolution
        else:  # 30% chance for anisotropic
            # Sample one dimension to be low resolution
            resolution = random.uniform(self.atlas_res, self.max_res_aniso)
            blur_res = resolution * self.thickness_factor
        
        return resolution, blur_res
    
    def _blurring_sigma_for_downsampling(self, atlas_res, target_res, thickness=None):
        """Calculate sigma for Gaussian blur to simulate resolution."""
        if thickness is None:
            thickness = target_res
        
        # Following SynthSeg's formula
        sigma = 0.75 * thickness / atlas_res
        return max(0, sigma)
    
    def __call__(self, data):
        d = dict(data)
        
        if random.random() < self.prob:
            # Sample resolution for this sample
            resolution, blur_res = self._sample_resolution()
            
            for key in self.keys:
                img = d[key]
                
                # Calculate blur sigma
                sigma = self._blurring_sigma_for_downsampling(
                    self.atlas_res, resolution, blur_res
                )
                
                if sigma > 0:
                    # Apply Gaussian blur to simulate lower resolution
                    img_np = img.cpu().numpy()
                    blurred = np.stack([
                        gaussian_filter(img_np[c], sigma=sigma)
                        for c in range(img_np.shape[0])
                    ], axis=0)
                    img = torch.from_numpy(blurred).to(img.device).type(img.dtype)
                
                # Optionally simulate actual downsampling/upsampling
                if resolution > self.atlas_res * 1.1:  # Only if significant difference
                    original_shape = img.shape[1:]  # Remove channel dim
                    
                    # Calculate downsampled size
                    downsample_factor = resolution / self.atlas_res
                    new_size = [max(1, int(s / downsample_factor)) for s in original_shape]
                    
                    # Downsample
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=new_size,
                        mode='trilinear',
                        align_corners=False,
                        antialias=False
                    ).squeeze(0)
                    
                    # Upsample back to original size
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=original_shape,
                        mode='trilinear',
                        align_corners=False
                    ).squeeze(0)
                
                d[key] = img
        
        return d


class IntensityClipNormalizeD(MapTransform):
    """
    Intensity clipping and normalization using percentiles.
    
    This transform:
    1. Clips intensities to 1st and 99th percentiles to remove outliers
    2. Normalizes to [0, 1] range
    3. Optionally applies gamma correction
    """
    
    def __init__(self,
                 keys,
                 clip_percentiles: tuple[float, float] = (1.0, 99.0),  # 1% and 99% percentiles
                 normalise: bool = True,
                 gamma_std: float = 0.5,
                 separate_channels: bool = True,
                 prob: float = 0.95):
        super().__init__(keys)
        self.clip_percentiles = clip_percentiles
        self.normalise = normalise
        self.gamma_std = gamma_std
        self.separate_channels = separate_channels
        self.prob = prob
    
    def __call__(self, data):
        d = dict(data)
        
        if random.random() < self.prob:
            for key in self.keys:
                img = d[key]
                
                if self.separate_channels:
                    # Process each channel separately
                    for c in range(img.shape[0]):
                        channel = img[c]
                        
                        # Clip intensities using percentiles
                        low_percentile, high_percentile = self.clip_percentiles
                        min_val = torch.quantile(channel, low_percentile / 100.0)
                        max_val = torch.quantile(channel, high_percentile / 100.0)
                        
                        # Ensure we don't have min_val == max_val
                        if max_val <= min_val:
                            max_val = min_val + 1e-8
                        
                        # Clip to percentile range
                        channel = torch.clamp(channel, min_val, max_val)
                        
                        # Normalize to [0, 1]
                        if self.normalise:
                            channel = (channel - min_val) / (max_val - min_val)
                        
                        # Apply gamma correction
                        if self.gamma_std > 0:
                            gamma = torch.exp(torch.randn(1) * self.gamma_std).item()
                            channel = channel.pow(gamma)
                        
                        img[c] = channel
                else:
                    # Process all channels together
                    low_percentile, high_percentile = self.clip_percentiles
                    min_val = torch.quantile(img, low_percentile / 100.0)
                    max_val = torch.quantile(img, high_percentile / 100.0)
                    
                    # Ensure we don't have min_val == max_val
                    if max_val <= min_val:
                        max_val = min_val + 1e-8
                    
                    # Clip to percentile range
                    img = torch.clamp(img, min_val, max_val)
                    
                    if self.normalise:
                        img = (img - min_val) / (max_val - min_val)
                    
                    if self.gamma_std > 0:
                        gamma = torch.exp(torch.randn(1) * self.gamma_std).item()
                        img = img.pow(gamma)
                
                d[key] = img
        
        return d
    


# -------------------------------------------------------------------------
# ConvertLabelsD
# -------------------------------------------------------------------------
from monai.transforms import MapTransform
from typing import Sequence, Mapping, Hashable, Dict, Union
import numpy as np
import torch

class ConvertLabelsD(MapTransform):
    """
    Replace integer labels of a segmentation **in-place**:

        every voxel with value  generation_labels[i]
        → becomes             output_labels[i]

    Parameters
    ----------
    keys : str | Sequence[str]
        Dict keys that should be remapped (usually just ["image"] or ["label"]).
    generation_labels : Sequence[int]
        Source label values (must be unique).
    output_labels : Sequence[int]
        Target label values (same length & order as generation_labels).
    background_label : int
        Fallback value for voxels whose label is *not* in generation_labels.
    """
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        generation_labels: Sequence[int],
        output_labels: Sequence[int],
        background_label: int = 0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

        gen = np.asarray(generation_labels, dtype=np.int64)
        out = np.asarray(output_labels,    dtype=np.int64)
        if gen.shape != out.shape:
            raise ValueError(
                f"`generation_labels` and `output_labels` must have same length "
                f"(got {gen.shape} vs {out.shape})."
            )

        # build a vectorised look-up table for fast mapping
        self._lut_size = int(gen.max()) + 1
        lut = np.full(self._lut_size, background_label, dtype=np.int64)
        lut[gen] = out
        self._lut = torch.from_numpy(lut)  # 1-D tensor, lives on CPU

    # ------------------------------------------------------------------ #
    def _convert(self, arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Vectorised remap – works for numpy or torch arrays."""
        if isinstance(arr, np.ndarray):
            if arr.max() >= self._lut_size:
                raise ValueError("Label value out of LUT bounds – enlarge generation_labels.")
            return self._lut.numpy()[arr]          # fancy-indexing on numpy array
        elif isinstance(arr, torch.Tensor):
            if arr.max() >= self._lut_size:
                raise ValueError("Label value out of LUT bounds – enlarge generation_labels.")
            # Move LUT to the same device as the input tensor
            lut_device = self._lut.to(arr.device)
            return lut_device[arr.long()]
        else:
            raise TypeError("Unsupported array type, must be numpy.ndarray or torch.Tensor")

    # ------------------------------------------------------------------ #
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:  # noqa: D401
        d = dict(data)
        for k in self.key_iterator(d):
            d[k] = self._convert(d[k])
        return d