"""
B A - D a t a s e t
Light-weight dataset for 3-D brain age prediction.

Changes vs. previous version
────────────────────────────
1.  No in-process cache by default  →  huge RAM saving with many workers
2.  Optional SMALL shared cache (multiprocessing.Manager) for “hot”
    samples when you really need it.
3.  Identical public interface: __len__, __getitem__, arguments.
"""

from __future__ import annotations
from pathlib import Path
from typing   import List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------- #
#                               shared cache                            #
# --------------------------------------------------------------------- #
import multiprocessing as mp
_MANAGER = mp.Manager()                           # created once per job
_SHARED_CACHE: Dict[int, np.ndarray] = _MANAGER.dict()

__all__ = ["BADataset"]


class BADataset(Dataset):
    """
    Parameters
    ----------
    file_paths   : list of paths to .npy volumes
    age_labels   : same length list/array of float ages
    sample_wts   : optional per-sample weights
    transform    : callable transform to run *in the worker / CPU*
    cache_size   : 0  → no per-worker cache  (recommended)
                   >0 → per-worker LRU cache of that many samples
    shared_cache : 0  → no shared cache
                   >0 → a *global* LRU cache (common to all workers)
    mode         : 'train' | 'val' | 'test'  (apply transform only in train)
    """

    def __init__(
        self,
        file_paths   : List[str | Path],
        age_labels   : List[float],
        sample_wts   : Optional[List[float]] = None,
        transform    = None,
        cache_size   : int = 0,        # 0 ⇒ off
        shared_cache : int = 0,        # 0 ⇒ off
        mode         : str = "train",
    ):
        assert len(file_paths) == len(age_labels), "len(paths) ≠ len(labels)"
        self.file_paths    = [str(p) for p in file_paths]
        self.age_labels    = age_labels
        self.sample_wts    = sample_wts
        self.transform     = transform
        self.mode          = mode.lower()

        # --- local per-process cache (OrderedDict) ---------------------- #
        from collections import OrderedDict
        self.cache_size    = max(0, cache_size)
        self._cache        : Dict[int, np.ndarray] = OrderedDict()

        # --- optional small shared cache ------------------------------- #
        self.shared_cap    = max(0, shared_cache)   # 0 → disabled

    # ------------------------------------------------------------------ #
    #                           internal I/O                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_volume(path: str) -> np.ndarray:
        # np.load is ~2× faster than nibabel for .npy data
        return np.load(path)           # (D,H,W)  dtype=float32

    # ------------------------------------------------------------------ #
    #                         torch Dataset API                           #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ---- 1. try shared cache (fast path) -------------------------- #
        if self.shared_cap and idx in _SHARED_CACHE:
            img_np = _SHARED_CACHE[idx]

        # ---- 2. try per-process LRU cache ----------------------------- #
        elif self.cache_size and idx in self._cache:
            img_np = self._cache.pop(idx)          # LRU refresh

        # ---- 3. load from disk --------------------------------------- #
        else:
            img_np = self._load_volume(self.file_paths[idx])

            # add to per-process cache
            if self.cache_size:
                # evict oldest if full
                if len(self._cache) >= self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[idx] = img_np

            # add to shared cache (bounded)
            if self.shared_cap and len(_SHARED_CACHE) < self.shared_cap:
                _SHARED_CACHE[idx] = img_np

        # ---- 4. build sample dict ------------------------------------ #
        sample = {
            "image": torch.from_numpy(img_np).unsqueeze(0),   # (1,D,H,W)
            "age"  : torch.tensor(self.age_labels[idx], dtype=torch.float32),
        }
        if self.sample_wts is not None:
            sample["weight"] = torch.tensor(self.sample_wts[idx],
                                            dtype=torch.float32)

        # ---- 5. transform (train only) ------------------------------- #
        if self.transform is not None and self.mode == "train":
            sample = self.transform(sample)

        return sample