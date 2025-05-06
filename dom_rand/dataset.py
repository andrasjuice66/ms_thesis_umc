"""
Light-weight Dataset with LRU in-RAM cache + per-sample weight support.
"""

from __future__ import annotations
import random
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio          # used only if you decide to add heavy IO aug

__all__ = ["BADataset"]


class BADataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        age_labels: List[float],
        sample_wts: Optional[List[float]] = None,
        transform = None,
        cache_size: int = 128,
        target_shape: Tuple[int,int,int] = (176,240,256),
        normalize: bool = True,
        mode: str = "train",
    ):
        self.file_paths  = file_paths
        self.age_labels  = age_labels
        self.sample_wts  = sample_wts
        self.transform   = transform
        self.cache_size  = cache_size
        self.cache       : Dict[int,np.ndarray] = OrderedDict()
        self.target_shape= target_shape
        self.normalize   = normalize
        self.mode        = mode.lower()
        assert len(self.file_paths) == len(self.age_labels)

    # ───────────────────────── internals ───────────────────────── #
    def _load_image(self, idx: int) -> np.ndarray:
        img = nib.load(self.file_paths[idx]).get_fdata(dtype=np.float32)
        if self.normalize:
            vmin, vmax = np.percentile(img, (1, 99))
            img = np.clip(img, vmin, vmax)
            img = (img - vmin) / (vmax - vmin + 1e-6)
        # (optional) resize / crop to target_shape could be added here
        return img

    # ───────────────────────── Dataset API ─────────────────────── #
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self.cache:
            img = self.cache.pop(idx)
        else:
            img = self._load_image(idx)
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
        self.cache[idx] = img
        sample = {
            "image": torch.from_numpy(img).unsqueeze(0),            # (1,D,H,W)
            "age"  : torch.tensor(self.age_labels[idx], dtype=torch.float32),
        }
        if self.sample_wts is not None:
            sample["weight"] = torch.tensor(self.sample_wts[idx], dtype=torch.float32)

        if self.transform is not None and self.mode == "train":
            sample = self.transform(sample)

        return sample