#!/usr/bin/env python
"""
Inference & 3-regime evaluation for an AGE-BIN classifier (e.g. SFCN).

Regimes
-------
1. Normal test
2. Domain-randomised test   (10 folds)
3. Dom-rand + tumour sim    (10 folds)

Ensemble
--------
5 checkpoints → median fusion → brain-age correction.

Author: <you>
Date  : 2025-06-09
"""

# -------- imports ----------------------------------------------------------
import os, sys, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import wandb                                    # comment out if not needed

import torchio
from monai.data import CacheDataset
from monai.transforms import (
    Compose, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd, MapTransform
)

# project imports – keep identical to your existing tree
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer
from brain_age_pred.training.metrics import calculate_metrics
from brain_age_pred.utils.utils import read_csv

# ---------- CONFIG ---------------------------------------------------------

# --- ❶ bin settings (MUST match training) ----------------------------------
BIN_RANGE = (42, 82)   # inclusive range used to build bins
BIN_STEP  = 1
bin_centres = np.arange(BIN_RANGE[0] + BIN_STEP / 2,
                        BIN_RANGE[1] + BIN_STEP / 2,
                        BIN_STEP, dtype=np.float32)
N_BINS = len(bin_centres)

# --- ❷ paths ---------------------------------------------------------------
MODEL_DIR   = '/home/ajoos/model_files/'
MODEL_PATH = os.path.join(MODEL_DIR, 'sfcn_original_ckp.pth')

TEST_CSV    = '/home/ajoos/brain_age_pred/data/labels/test.csv'
DATA_ROOT   = '/scratch-shared/ajoos/'

OUT_DIR     = Path('.')
OUT_DIR.mkdir(exist_ok=True)

# --- ❸ runtime -------------------------------------------------------------
BATCH_SIZE  = 1
NUM_WORKERS = 4
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WAND  = True          # switch off if you do not want Weights&Biases
WANDB_API = '2abdb867a9244072f2237704a3cacc77fa548dd8'

# ---------------------------------------------------------------------------


# ======================= MODEL ============================================


class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        return out




class SFCN_Bins(nn.Module):
    """
    Thin wrapper around the SFCN (or any classifier with age bins).

    forward_logits(): raw logits  (B, N_BINS)
    forward()       : expected age (B,)
    """
    def __init__(self):
        super().__init__()
        self.net = SFCN(output_dim=N_BINS)

    def forward_logits(self, x):
        # original SFCN returns a list [log_probs] of shape (B, N_BINS, 1,1,1)
        logp = self.net(x)[0]                 # (B, N_BINS, 1,1,1)
        logp = logp.squeeze()                 # (B, N_BINS)
        return logp

    def forward(self, x):
        logp = self.forward_logits(x)
        p    = torch.exp(logp)                # softmax already inside SFCN
        # dot product with bin centres
        age  = (p * torch.tensor(bin_centres, device=x.device)).sum(dim=1)
        return age                            # (B,)


def load_model_chkpt(chkpt_path, device=DEVICE):
    model = SFCN_Bins().to(device)
    state = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ======================= DATA PIPELINE ====================================

class LoadNpy(MapTransform):
    """MONAI transform: load .npy file & add channel dim if absent."""
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            arr = np.load(d[k]).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[None]       # C=1
            d[k] = arr
        return d


def monai_transforms():
    x, y, z = (160, 192, 160)
    p       = 1.0   # isotropic spacing wanted
    return Compose([
        LoadNpy(keys=['image']),
        Spacingd(keys=['image'], pixdim=(p, p, p)),
        CropForegroundd(keys=['image'], source_key='image', allow_smaller=True),
        SpatialPadd(keys=['image'], spatial_size=(x, y, z)),
        CenterSpatialCropd(keys=['image'], roi_size=(x, y, z)),
        torchio.transforms.ZNormalization(masking_method=lambda im: im > 0,
                                           keys=['image'], include=['image']),
    ])


def dataloader_from_csv(csv_path, data_root, batch_size=BATCH_SIZE):
    df = pd.read_csv(csv_path).dropna(subset=['image_path', 'age'])
    records = []
    for _, r in df.iterrows():
        p = r['image_path']
        p = os.path.join(data_root, p) if not os.path.isabs(p) else p
        records.append({'image': p, 'label': r['age']})
    ds = CacheDataset(records, transform=monai_transforms(),
                      cache_rate=0.2, num_workers=NUM_WORKERS)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    pin_memory=torch.cuda.is_available(),
                    num_workers=NUM_WORKERS)
    return dl, df


# ======================= EVALUATION HELPERS ===============================

def brain_age_correction(pred, ca):
    """Same rule as the regression script."""
    return np.where(ca > 18, pred + (ca * 0.062) - 2.96, pred)


def predict_single_model(model_path, loader):
    """Predictions using a single model."""
    print(f'▶ Loading model: {Path(model_path).name}')
    model = load_model_chkpt(model_path)
    
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].to(DEVICE)
            label = batch['label'].cpu().numpy()
            out = model(img)  # (B,)
            preds.append(out.cpu().numpy())
            targets.append(label)
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    torch.cuda.empty_cache()
    return preds, targets


def single_eval(loader, label_array, regime):
    preds, t = predict_single_model(MODEL_PATH, loader)
    assert np.allclose(t, label_array)
    preds_corr = brain_age_correction(preds, t)
    metrics = calculate_metrics(preds_corr, t, modalities=None, sexes=None)
    print(f'{regime}: MAE={metrics["mae"]:.3f},  R²={metrics["r2"]:.3f}')
    return metrics, preds, preds_corr, t


# --------------- domain randomisation helpers -----------------------------

def make_domrand_transform(device, use_tumor=False):
    cfg = {
        "use_domain_randomization": True,
        "transform_probs": {
            # same probabilities you used before
            "flip": 0.5, "affine": 0.8, "contrast": 0.6, "gamma": 0.5,
            "blur": 0.4, "bias": 0.5, "scale_int": 0.4, "shift_int": 0.4,
            "hist_shift": 0.3, "noise": 0.4, "rician": 0.3, "gibbs": 0.3,
            "resolution": 0.5, "coarse_do": 0.3, "crop": 1.0,
            "tumor": 0.3 if use_tumor else 0.0,
        },
        "output_shape": (160, 192, 160),
    }
    tumor_cfg = {
        "use_tumor_simulation": use_tumor,
        "prob": 0.3, "use_age_based_segmentation": False,
        "perlin_res": [2, 2, 2], "tumor_size_factor_range": [0.5, 2.0],
        "use_fluid_dynamics": True,
    } if use_tumor else {}
    return DomainRandomizer(device=device, **cfg, tumor_config=tumor_cfg)


def create_domrand_loader(csv_path, data_root, transform, batch_size=BATCH_SIZE):
    fp, ages, sw, sexes, mods = read_csv(csv_path, data_root)
    ds = BADataset(fp, ages, sexes, mods, transform=transform, mode='test', cache_size=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    return dl, ages, sexes, mods


# ======================= 3-REGIME RUN =====================================

def three_regime_evaluation():
    if WAND:
        wandb.login(key=WANDB_API)
        wandb.init(project='brainage-bins', name=f'sfcn_bins_{datetime.now():%Y%m%d_%H%M%S}',
                   config=dict(model_path=MODEL_PATH, bins=N_BINS))
    # ---------- 1. normal --------------------------------------------------
    norm_loader, norm_df = dataloader_from_csv(TEST_CSV, DATA_ROOT)
    norm_metrics, norm_raw, norm_corr, tgt = single_eval(norm_loader, norm_df['age'].values, 'Normal')

    if WAND: wandb.log({f"normal/{k}": v for k, v in norm_metrics.items()})

    # ---------- 2. domain-randomised 10× -----------------------------------
    dom_metrics_all = []
    domrand_tf = make_domrand_transform(DEVICE, use_tumor=False)
    for i in range(10):
        print(f'-- Domain-rand  fold {i+1}/10')
        dom_loader, ages, *_ = create_domrand_loader(TEST_CSV, DATA_ROOT, domrand_tf)
        m, *_ = single_eval(dom_loader, np.array(ages), f'DomRand_{i}')
        dom_metrics_all.append(m)
    dom_avg = {k: np.mean([m[k] for m in dom_metrics_all]) for k in dom_metrics_all[0]}
    dom_std = {k+'_std': np.std([m[k] for m in dom_metrics_all]) for k in dom_metrics_all[0]}
    dom_metrics = {**dom_avg, **dom_std}
    if WAND: wandb.log({f"domrand/{k}": v for k, v in dom_metrics.items()})

    # ---------- 3. dom-rand + tumour 10× -----------------------------------
    tum_metrics_all = []
    tum_tf = make_domrand_transform(DEVICE, use_tumor=True)
    for i in range(10):
        print(f'-- DomRand+Tumour  fold {i+1}/10')
        tum_loader, ages, *_ = create_domrand_loader(TEST_CSV, DATA_ROOT, tum_tf)
        m, *_ = single_eval(tum_loader, np.array(ages), f'DomRandTum_{i}')
        tum_metrics_all.append(m)
    tum_avg = {k: np.mean([m[k] for m in tum_metrics_all]) for k in tum_metrics_all[0]}
    tum_std = {k+'_std': np.std([m[k] for m in tum_metrics_all]) for k in tum_metrics_all[0]}
    tum_metrics = {**tum_avg, **tum_std}
    if WAND: wandb.log({f"domrand_tumour/{k}": v for k, v in tum_metrics.items()})

    # ---------- summary & save --------------------------------------------
    summary = dict(normal=norm_metrics, dom_rand=dom_metrics, dom_rand_tum=tum_metrics)
    (OUT_DIR/'sfcn_bins_3regimes_results.json').write_text(json.dumps(summary, indent=2))

    print('\n=== SUMMARY ===')
    for k in summary:
        print(f'{k:14}: MAE={summary[k]["mae"]:.3f}  R²={summary[k]["r2"]:.3f}')

    if WAND:
        wandb.finish()
    return summary


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    three_regime_evaluation()