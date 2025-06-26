# brain_age_pred/dataset/custom_transformations.py

import torch
import numpy as np
from monai.transforms import Transform
import numpy as np
import torch
from typing import Dict, Hashable, Mapping
from monai.config import KeysCollection
from monai.transforms import MapTransform, Transform


class SampleConditionalGMMd(Transform):
    """
    Given a segmentation map (integer labels), sample a synthetic image by:
      1) drawing per‐label means & stds from hyperpriors,
      2) sampling each voxel intensity ~ Normal( mean[label], std[label] ).
    """
    def __init__(
        self,
        seg_key: str = "image",
        out_key: str = "image",
        prior_means: np.ndarray = None,
        prior_stds:  np.ndarray = None,
        distribution: str = "normal",  # "normal" or "uniform"
    ):
        """
        prior_means: shape (2, n_classes), row 0 = loc for GMM means,
                                          row 1 = scale (std) for their hyper‐prior
        prior_stds:  shape (2, n_classes), row 0 = loc for GMM stds,
                                          row 1 = scale for their hyper‐prior
        """
        self.seg_key = seg_key
        self.out_key = out_key
        assert distribution in ("normal", "uniform")
        self.distribution = distribution

        if prior_means is None or prior_stds is None:
            raise ValueError("Must provide `prior_means` and `prior_stds` arrays")

        pm = np.asarray(prior_means, dtype=float)
        ps = np.asarray(prior_stds,  dtype=float)
        assert pm.shape[0] == 2 and ps.shape[0] == 2
        assert pm.shape == ps.shape

        # hyper‐prior parameters
        self.hyper_mean_loc   = pm[0]
        self.hyper_mean_scale = pm[1]
        self.hyper_std_loc    = ps[0]
        self.hyper_std_scale  = ps[1]

    def __call__(self, data):
        # segmentation tensor
        seg = data[self.seg_key]  # shape [1, D, H, W] or [C, D, H, W]
        # drop channel‐first if it's size 1
        if seg.dim() == 4 and seg.shape[0] == 1:
            seg_s = seg.squeeze(0)
        else:
            seg_s = seg
        device = seg.device

        # sample GMM parameters - FORCE FLOAT32
        loc_means  = torch.from_numpy(self.hyper_mean_loc).to(device, dtype=torch.float32)
        scale_means= torch.from_numpy(self.hyper_mean_scale).to(device, dtype=torch.float32)
        loc_stds   = torch.from_numpy(self.hyper_std_loc).to(device, dtype=torch.float32)
        scale_stds = torch.from_numpy(self.hyper_std_scale).to(device, dtype=torch.float32)

        if self.distribution == "normal":
            gmm_means = torch.normal(loc_means,  scale_means)
            gmm_stds  = torch.normal(loc_stds,   scale_stds)
        else:  # uniform
            low_m  = loc_means - scale_means
            high_m = loc_means + scale_means
            gmm_means = low_m + (high_m - low_m) * torch.rand_like(loc_means)
            low_s  = loc_stds  - scale_stds
            high_s = loc_stds  + scale_stds
            gmm_stds  = low_s  + (high_s  - low_s)  * torch.rand_like(loc_stds)

        # force positive std
        gmm_stds = (gmm_stds.abs() + 1e-6)

        # flatten segmentation
        flat_labels = seg_s.flatten().long()
        # map per‐voxel means/stds
        mean_map = gmm_means[flat_labels]
        std_map  = gmm_stds[flat_labels]
        # sample intensities
        intens_flat = torch.normal(mean_map, std_map)
        intens = intens_flat.view(seg_s.shape)

        # restore channel dim
        intens = intens.unsqueeze(0)

        data[self.out_key] = intens
        return data
    

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
