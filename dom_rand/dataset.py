"""
B A - D a t a s e t
Light-weight dataset for 3-D brain age prediction.

Changes vs. previous version
────────────────────────────
1.  Removed shared cache to avoid multiprocessing issues
2.  Kept optional per-worker LRU cache for performance
3.  Identical public interface: __len__, __getitem__, arguments.
"""

from __future__ import annotations
from pathlib import Path
from typing   import List, Optional, Dict
from collections import OrderedDict
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["BADataset"]


class BADataset(Dataset):
    """
    Parameters
    ----------
    file_paths   : list of paths to .npy volumes
    age_labels   : same length list/array of float ages
    modalities   : optional list of modalities per sample
    sexes        : optional list of sexes per sample
    sample_wts   : optional per-sample weights
    transform    : callable transform to run *in the worker / CPU*
    cache_size   : 0  → no per-worker cache  (recommended)
                   >0 → per-worker LRU cache of that many samples
    mode         : 'train' | 'val' | 'test'  (apply transform only in train)
    """

    def __init__(
        self,
        file_paths   : List[str | Path],
        age_labels   : List[float],
        modalities   : Optional[List[str]] = None,
        sexes        : Optional[List[str]] = None,
        sample_wts   : Optional[List[float]] = None,
        transform    = None,
        cache_size   : int = 0,        # 0 ⇒ off
        mode         : str = "train",
    ):
        assert len(file_paths) == len(age_labels), "len(paths) ≠ len(labels)"
        if modalities is not None:
            assert len(modalities) == len(file_paths), "len(modalities) ≠ len(paths)"
        if sexes is not None:
            assert len(sexes) == len(file_paths), "len(sexes) ≠ len(paths)"
        self.file_paths    = [str(p) for p in file_paths]
        self.age_labels    = age_labels
        self.modalities    = modalities
        self.sexes         = sexes
        self.sample_wts    = sample_wts
        self.transform     = transform
        self.mode          = mode.lower()

        # --- local per-process cache (OrderedDict) ---------------------- #
        self.cache_size    = max(0, cache_size)
        self._cache        : Dict[int, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------ #
    #                           internal I/O                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_volume(path: str) -> np.ndarray:
        return np.load(path)           # (D,H,W)  dtype=float32

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img_np = self._load_volume(self.file_paths[idx])

        sample = {
            "image": torch.from_numpy(img_np).unsqueeze(0),
            "age":   torch.tensor(self.age_labels[idx], dtype=torch.float32),
        }
        if self.sample_wts is not None:
            sample["weight"] = torch.tensor(self.sample_wts[idx], dtype=torch.float32)
        if self.modalities is not None:
            sample["modality"] = self.modalities[idx]
        if self.sexes is not None:
            sample["sex"] = self.sexes[idx]

        # ---- transform -------------------------------------------------
        if self.transform is not None and self.mode == "train":
            sample = self.transform(sample)

        # ---- sanity checks --------------------------------------------
        if sample is None:
            raise RuntimeError(f"Transform returned None for idx {idx}")
        if sample.get("image") is None:
            raise RuntimeError(f"Image is None for idx {idx}")

        return sample