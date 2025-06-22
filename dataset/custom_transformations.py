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


class HemisphereAwareFlipD(MapTransform):
    """
    Hemisphere-aware random flipping with label swapping following SynthSeg logic.
    
    This transform:
    1. Flips the image/labels along the left-right axis (typically axis 0)
    2. Swaps corresponding left-right anatomical labels to maintain anatomical consistency
    3. Leaves neutral (non-sided) labels unchanged
    """
    
    def __init__(self,
                 keys,
                 generation_labels: np.ndarray = None,
                 n_neutral_labels: int = 7,
                 spatial_axis: int = 0,  # Left-right axis
                 prob: float = 0.5):
        super().__init__(keys)
        self.generation_labels = generation_labels
        self.n_neutral_labels = n_neutral_labels
        self.spatial_axis = spatial_axis
        self.prob = prob
        
        # Create label swapping lookup table if labels are provided
        if generation_labels is not None:
            self.swap_lut = self._create_swap_lut()
        else:
            self.swap_lut = None
    
    def _create_swap_lut(self):
        """Create lookup table for swapping left-right labels."""
        n_labels = len(self.generation_labels)
        
        # If all labels are neutral, no swapping needed
        if self.n_neutral_labels >= n_labels:
            return None
        
        # Split labels: neutral, left hemisphere, right hemisphere
        neutral = self.generation_labels[:self.n_neutral_labels]
        n_sided = (n_labels - self.n_neutral_labels) // 2
        left = self.generation_labels[self.n_neutral_labels:self.n_neutral_labels + n_sided]
        right = self.generation_labels[self.n_neutral_labels + n_sided:]
        
        # Create mapping for label swapping (neutral stays, left<->right swap)
        swapped_labels = np.concatenate([neutral, right, left])
        
        # Create lookup table that maps each original label to its swapped version
        max_label = max(np.max(self.generation_labels), np.max(swapped_labels))
        lut = np.arange(max_label + 1)  # Identity mapping by default
        
        for orig, swap in zip(self.generation_labels, swapped_labels):
            lut[orig] = swap
            
        return lut
    
    def __call__(self, data):
        d = dict(data)
        
        if random.random() < self.prob:
            for key in self.keys:
                # Flip along the specified spatial axis
                # Add 1 to account for channel dimension in MONAI format (C,H,W,D)
                flip_axis = self.spatial_axis + 1
                d[key] = torch.flip(d[key], dims=[flip_axis])
                
                # Apply label swapping if this is a segmentation and we have labels
                if self.swap_lut is not None and key == self.keys[0]:  # Assume first key is segmentation
                    # Convert to numpy for indexing, then back to tensor
                    seg = d[key].cpu().numpy().astype(int)
                    # Apply lookup table to swap labels
                    seg_swapped = self.swap_lut[np.clip(seg, 0, len(self.swap_lut)-1)]
                    d[key] = torch.from_numpy(seg_swapped).to(d[key].device).type(d[key].dtype)
        
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
                        antialias=True
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
    Intensity clipping and normalization following SynthSeg's IntensityAugmentation.
    
    This transform:
    1. Clips intensities to remove outliers
    2. Normalizes to [0, 1] range
    3. Optionally applies gamma correction
    """
    
    def __init__(self,
                 keys,
                 clip_value: float = 300.0,
                 normalise: bool = True,
                 gamma_std: float = 0.5,
                 separate_channels: bool = True,
                 prob: float = 0.95):
        super().__init__(keys)
        self.clip_value = clip_value
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
                        
                        # Clip intensities
                        if self.clip_value > 0:
                            channel = torch.clamp(channel, 0, self.clip_value)
                        
                        # Normalize to [0, 1]
                        if self.normalise:
                            min_val = channel.min()
                            max_val = channel.max()
                            if max_val > min_val:
                                channel = (channel - min_val) / (max_val - min_val)
                        
                        # Apply gamma correction
                        if self.gamma_std > 0:
                            gamma = torch.exp(torch.randn(1) * self.gamma_std).item()
                            channel = channel.pow(gamma)
                        
                        img[c] = channel
                else:
                    # Process all channels together
                    if self.clip_value > 0:
                        img = torch.clamp(img, 0, self.clip_value)
                    
                    if self.normalise:
                        min_val = img.min()
                        max_val = img.max()
                        if max_val > min_val:
                            img = (img - min_val) / (max_val - min_val)
                    
                    if self.gamma_std > 0:
                        gamma = torch.exp(torch.randn(1) * self.gamma_std).item()
                        img = img.pow(gamma)
                
                d[key] = img
        
        return d
    

# brain_age_pred/brain_gen/synthseg_transforms.py

import numpy as np
import torch
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union
from monai.config import KeysCollection
from monai.transforms import MapTransform, Transform
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F


class MultiChannelSampleConditionalGMMd(MapTransform):
    """
    Multi-channel version of SampleConditionalGMMd that generates different 
    intensities per channel, following SynthSeg's approach.
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        seg_key: str = "image",
        out_key: str = "image", 
        prior_means: np.ndarray = None,
        prior_stds: np.ndarray = None,
        distribution: str = "uniform",
        n_channels: int = 1,
        use_specific_stats_for_channel: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.seg_key = seg_key
        self.out_key = out_key
        self.prior_means = prior_means
        self.prior_stds = prior_stds
        self.distribution = distribution
        self.n_channels = n_channels  
        self.use_specific_stats_for_channel = use_specific_stats_for_channel
        
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        seg = d[self.seg_key]
        
        if self.n_channels == 1:
            # Single channel - use existing logic
            from brain_age_pred.brain_gen.gen_image_from_labels import SampleConditionalGMMd
            single_channel_transform = SampleConditionalGMMd(
                seg_key=self.seg_key,
                out_key=self.out_key,
                prior_means=self.prior_means,
                prior_stds=self.prior_stds,
                distribution=self.distribution,
            )
            return single_channel_transform(d)
        
        # Multi-channel generation
        channels = []
        for ch in range(self.n_channels):
            # Select channel-specific stats if available
            if self.use_specific_stats_for_channel and self.prior_means.shape[0] >= self.n_channels:
                channel_means = self.prior_means[ch::self.n_channels]  # Every n_channels row
                channel_stds = self.prior_stds[ch::self.n_channels]
            else:
                # Use same stats for all channels (with some randomization)
                channel_means = self.prior_means
                channel_stds = self.prior_stds
            
            # Generate channel using single channel transform
            single_channel_data = {self.seg_key: seg}
            single_channel_transform = SampleConditionalGMMd(
                seg_key=self.seg_key,
                out_key=self.out_key,
                prior_means=channel_means,
                prior_stds=channel_stds,
                distribution=self.distribution,
            )
            
            channel_result = single_channel_transform(single_channel_data)
            channels.append(channel_result[self.out_key])
        
        # Stack channels
        if len(channels) > 1:
            d[self.out_key] = torch.cat(channels, dim=-1)  # Concatenate along channel dimension
        else:
            d[self.out_key] = channels[0]
            
        return d


class ConvertLabelsD(MapTransform):
    """
    Convert generation labels to output labels following SynthSeg's approach.
    Maps label values from one set to another (e.g., to remove certain structures).
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        generation_labels: np.ndarray,
        output_labels: np.ndarray,
        background_label: int = 0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.generation_labels = generation_labels
        self.output_labels = output_labels
        self.background_label = background_label
        
        # Create mapping dictionary
        self.label_mapping = {}
        for gen_label, out_label in zip(generation_labels, output_labels):
            self.label_mapping[int(gen_label)] = int(out_label)
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        for key in self.key_iterator(d):
            labels = d[key]
            
            # Convert to numpy for easier label mapping
            if isinstance(labels, torch.Tensor):
                was_tensor = True
                device = labels.device
                labels_np = labels.cpu().numpy()
            else:
                was_tensor = False
                labels_np = labels
            
            # Create output array filled with background
            output_labels_np = np.full_like(labels_np, self.background_label)
            
            # Apply mapping
            for gen_label, out_label in self.label_mapping.items():
                mask = labels_np == gen_label
                output_labels_np[mask] = out_label
            
            # Convert back if it was a tensor
            if was_tensor:
                d[key] = torch.tensor(output_labels_np, device=device, dtype=labels.dtype)
            else:
                d[key] = output_labels_np
                
        return d


class ImageGradientsD(MapTransform):
    """
    Compute spatial gradients of images using Sobel filters, following SynthSeg's approach.
    Returns the magnitude of the gradient.
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        method: str = "sobel",
        normalize: bool = True,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.method = method
        self.normalize = normalize
        
    def _sobel_gradients_3d(self, img: torch.Tensor) -> torch.Tensor:
        """Compute 3D Sobel gradients."""
        # Sobel kernels for 3D
        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
        
        sobel_z = torch.tensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
        
        # Apply convolutions
        grad_x = F.conv3d(img.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
        grad_y = F.conv3d(img.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)  
        grad_z = F.conv3d(img.unsqueeze(0).unsqueeze(0), sobel_z, padding=1)
        
        # Compute magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return gradient_magnitude.squeeze(0).squeeze(0)
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        for key in self.key_iterator(d):
            img = d[key]
            
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            
            if self.method == "sobel":
                if img.ndim == 3:  # 3D image
                    gradient_mag = self._sobel_gradients_3d(img)
                else:
                    raise NotImplementedError("Only 3D Sobel gradients implemented")
            else:
                raise ValueError(f"Unknown gradient method: {self.method}")
            
            if self.normalize:
                # Normalize to [0, 1] range
                gradient_mag = gradient_mag / (gradient_mag.max() + 1e-8)
            
            d[key] = gradient_mag
            
        return d


class RandomCropWithPaddingD(MapTransform):
    """
    Random crop with padding if necessary, following SynthSeg's approach.
    Ensures we can always get the desired crop size.
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        crop_size: Sequence[int],
        mode: str = "constant",
        constant_values: float = 0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.crop_size = ensure_tuple_rep(crop_size, 3)
        self.mode = mode
        self.constant_values = constant_values
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # Get current shape
            current_shape = img.shape
            
            # Calculate padding needed
            padding = []
            for i, (curr_size, crop_size) in enumerate(zip(current_shape, self.crop_size)):
                if curr_size < crop_size:
                    pad_total = crop_size - curr_size
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    padding.extend([pad_before, pad_after])
                else:
                    padding.extend([0, 0])
            
            # Apply padding if needed
            if any(p > 0 for p in padding):
                img = F.pad(img, padding, mode=self.mode, value=self.constant_values)
            
            # Random crop
            new_shape = img.shape
            starts = []
            for curr_size, crop_size in zip(new_shape, self.crop_size):
                if curr_size > crop_size:
                    max_start = curr_size - crop_size
                    start = torch.randint(0, max_start + 1, (1,)).item()
                else:
                    start = 0
                starts.append(start)
            
            # Extract crop
            if img.ndim == 3:
                cropped = img[
                    starts[0]:starts[0] + self.crop_size[0],
                    starts[1]:starts[1] + self.crop_size[1], 
                    starts[2]:starts[2] + self.crop_size[2]
                ]
            else:
                raise NotImplementedError("Only 3D cropping implemented")
            
            d[key] = cropped
            
        return d
