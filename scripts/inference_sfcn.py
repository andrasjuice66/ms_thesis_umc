#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFCN 3-fold inference / evaluation script
Fixed 2025-06-09
"""

# ────────────────────────────────── Imports ─────────────────────────────────
import io
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy.stats import norm
from torch.utils.data import DataLoader

# ─── Project-local modules ──────────────────────────────────────────────────
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer
from brain_age_pred.training.metrics import calculate_metrics
from brain_age_pred.utils.utils import read_csv

# ───────────────────────────────  Utilities  ───────────────────────────────
def safe_torch_load(fp, map_location="cpu"):
    """
    Load a PyTorch checkpoint while
    1) stripping an eventual UTF-8 BOM,
    2) forcing weights_only=False for PyTorch ≥ 2.6.
    """
    def _do_load(buffer):
        try:      # PyTorch ≥ 2.6
            return torch.load(buffer, map_location=map_location,
                              weights_only=False)
        except TypeError:  # PyTorch < 2.6 – no weights_only kwarg
            return torch.load(buffer, map_location=map_location)

    try:
        return _do_load(fp)
    except (pickle.UnpicklingError, RuntimeError):
        with open(fp, "rb") as f:
            raw = f.read()
        bom = b"\xef\xbb\xbf"
        if raw.startswith(bom):
            print("⚠️  UTF-8 BOM detected – stripping it and re-loading …")
            return _do_load(io.BytesIO(raw[len(bom):]))
        raise


def num2vect(x, bin_range, bin_step, sigma):
    """
    Map age(s) to hard/soft one-hot vector(s).

    Returns (vector, bin_centres)
    """
    start, end = bin_range
    bins = int((end - start) / bin_step)
    centres = start + bin_step * (0.5 + np.arange(bins))

    x = np.asarray(x).reshape(-1)
    if sigma == 0:                     # hard labels -> indices
        idx = ((x - start) // bin_step).astype(int)
        return idx, centres

    # soft labels
    v = np.empty((x.size, bins), dtype=np.float32)
    for j, age in enumerate(x):
        for i, c in enumerate(centres):
            x1, x2 = c - bin_step / 2, c + bin_step / 2
            v[j, i] = norm.cdf(x2, age, sigma) - norm.cdf(x1, age, sigma)
    return v.squeeze(), centres


def crop_center(vol, out_sp):
    """
    3-D / 4-D centre crop.
    """
    if vol.ndim not in (3, 4):
        raise ValueError(f"Expected 3-D or 4-D, got {vol.ndim}")
    dz = (vol.shape[-3] - out_sp[-3]) // 2
    dy = (vol.shape[-2] - out_sp[-2]) // 2
    dx = (vol.shape[-1] - out_sp[-1]) // 2
    if vol.ndim == 3:
        return vol[dz:-dz, dy:-dy, dx:-dx]
    return vol[:, dz:-dz, dy:-dy, dx:-dx]


def my_KLDivLoss(log_p, q):
    """
    Batch-wise KL divergence (averaged over batch).
    """
    q = q + 1e-16
    return F.kl_div(log_p, q, reduction="sum") / q.size(0)


# ───────────────────────────── Model definition ─────────────────────────────
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


# ───────────────────────────── Data utilities ───────────────────────────────
def create_test_dataloader(csv_path, data_dir, transform,
                           batch_size=8, workers=4):
    paths, ages, _, sexes, modalities = read_csv(csv_path, data_dir)
    ds = BADataset(
        file_paths=paths,
        age_labels=ages,
        sexes=sexes,
        modalities=modalities,
        transform=transform,
        mode="test",
        cache_size=0,
    )
    loader = DataLoader(ds, batch_size, False, num_workers=workers,
                        pin_memory=True)
    return loader, paths, ages, sexes, modalities


def create_eval_transforms(device, *, domain_rand=False, tumor=False):
    if not domain_rand:
        return None
    tf_cfg = {
        "use_domain_randomization": True,
        "transform_probs": {
            "flip": .5, "affine": .8, "contrast": .6, "gamma": .5,
            "blur": .4, "bias": .5, "scale_int": .4, "shift_int": .4,
            "hist_shift": .3, "noise": .4, "rician": .3, "gibbs": .3,
            "resolution": .5, "coarse_do": .3, "crop": 1.,
            "tumor": .3 if tumor else 0.,
        },
        "output_shape": (160, 192, 160),
    }
    tumor_cfg = {
        "use_tumor_simulation": tumor,
        "prob": .3, "use_age_based_segmentation": False,
        "perlin_res": [2, 2, 2],
        "tumor_size_factor_range": [0.5, 2.0],
        "use_fluid_dynamics": True,
    } if tumor else {}
    return DomainRandomizer(device=device,
                            use_tumor_simulation=tumor,
                            tumor_config=tumor_cfg,
                            **tf_cfg)


# ─────────────────────────── Evaluation functions ───────────────────────────
def run_single_evaluation(model, loader, device,
                          bin_range, bin_step, sigma, bin_centres):
    model.eval()
    bc = torch.tensor(bin_centres, device=device)
    preds, targs, losses = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            age_raw = batch["age"]            # still on CPU

            # ------- decide scalar vs soft-label -----------------------
            if age_raw.ndim == 2:             # already (B, nbins)
                soft = age_raw.to(device, dtype=torch.float32)
            else:                             # scalar ages -> build soft
                ages_np = age_raw.numpy()
                soft_np, _ = num2vect(ages_np, bin_range, bin_step, sigma)
                soft = torch.from_numpy(soft_np).to(device)

            targets_age = (soft * bc).sum(1)          # (B,)

            log_p = model(imgs)[0]
            losses.append(my_KLDivLoss(log_p, soft).item())

            pred = (torch.exp(log_p) * bc).sum(1)
            preds.append(pred.cpu().numpy())
            targs.append(targets_age.cpu().numpy())

    return (np.concatenate(preds),
            np.concatenate(targs),
            float(np.mean(losses)))


def run_multi_fold_evaluation(model, csv_path, data_dir, device, transform,
                              folds, tag,
                              bin_range, bin_step, sigma, bin_centres,
                              batch_size):
    print(f"Running {folds}-fold {tag} evaluation …")
    metrics_fold = []
    for k in range(folds):
        print(f"{tag} fold {k+1}/{folds}")
        loader, _, _, sexes, mods = create_test_dataloader(
            csv_path, data_dir, transform, batch_size)
        p, t, l = run_single_evaluation(
            model, loader, device, bin_range, bin_step, sigma, bin_centres)
        m = calculate_metrics(p, t, mods, sexes)
        m["loss"] = l
        metrics_fold.append(m)

    out = {}
    for key in metrics_fold[0]:
        v = [m[key] for m in metrics_fold]
        out[key] = float(np.mean(v))
        out[f"{key}_std"] = float(np.std(v))
    return out


# ────────────────────────── Main inference routine ──────────────────────────
def inference_with_3fold_evaluation():
    # ---- paths & options ---------------------------------------------------
    model_path = "/home/ajoos/model_files/sfcn_original_ckp.p"
    csv_test = "/home/ajoos/brain_age_pred/data/labels/test.csv"
    data_dir = "/scratch-shared/ajoos/"
    batch = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- wandb -------------------------------------------------------------
    wandb.login(key="2abdb867a9244072f2237704a3cacc77fa548dd8")
    run_name = f"sfcn_original_{datetime.now():%Y%m%d_%H%M%S}"
    wandb.init(project="brainage-inference", name=run_name,
               config=dict(model="SFCN", model_path=model_path,
                           test_csv=csv_test, batch_size=batch,
                           device=str(device)),
               reinit=True)

    print(f"Running on device: {device}")
    print(f"W&B run: {run_name}")

    try:
        # ---- model ---------------------------------------------------------
        model = SFCN()
        model = torch.nn.DataParallel(model)
        print(f"Loading model from {model_path}")
        state = safe_torch_load(model_path, map_location="cpu")
        sd = state["state_dict"] if (isinstance(state, dict)
                                     and "state_dict" in state) else state
        model.load_state_dict(sd)
        model.to(device).eval()
        print("✅ Model loaded successfully")

        # ---- bin setup -----------------------------------------------------
        bin_range = [42, 82]
        bin_step = 1
        sigma = 1
        bin_centres = bin_range[0] + bin_step * (
            0.5 + np.arange(int((bin_range[1] - bin_range[0]) / bin_step))
        )

        # ---- dataset-level logging ----------------------------------------
        fp, ages, _, sexes, mods = read_csv(csv_test, data_dir)
        wandb.log(dict(dataset_num=len(fp),
                       age_min=float(np.min(ages)),
                       age_max=float(np.max(ages)),
                       age_mean=float(np.mean(ages)),
                       age_std=float(np.std(ages))))

        # ---- 1) plain evaluation ------------------------------------------
        print("\n=== 1/3: Plain test evaluation ===")
        loader, _, _, _, _ = create_test_dataloader(
            csv_test, data_dir, None, batch)
        p_plain, t_plain, l_plain = run_single_evaluation(
            model, loader, device,
            bin_range, bin_step, sigma, bin_centres)
        m_plain = calculate_metrics(p_plain, t_plain, mods, sexes)
        m_plain["loss"] = l_plain
        print(f"Plain MAE={m_plain['mae']:.4f}, R²={m_plain['r2']:.4f}")
        wandb.log({f"plain_{k}": v for k, v in m_plain.items()})

        # ---- 2) domain randomization --------------------------------------
        print("\n=== 2/3: Domain-rand evaluation ===")
        tf_dom = create_eval_transforms(device, domain_rand=True, tumor=False)
        m_dom = run_multi_fold_evaluation(
            model, csv_test, data_dir, device, tf_dom, 10, "dom_rand",
            bin_range, bin_step, sigma, bin_centres, batch)
        wandb.log({f"dom_rand_{k}": v for k, v in m_dom.items()})

        # ---- 3) domain rand + tumor ---------------------------------------
        print("\n=== 3/3: Dom-rand + tumor evaluation ===")
        tf_dt = create_eval_transforms(device, domain_rand=True, tumor=True)
        m_dt = run_multi_fold_evaluation(
            model, csv_test, data_dir, device, tf_dt, 10, "dom_rand_tumor",
            bin_range, bin_step, sigma, bin_centres, batch)
        wandb.log({f"dom_rand_tumor_{k}": v for k, v in m_dt.items()})

        # ---- save / plot ---------------------------------------------------
        out = dict(plain=m_plain, dom_rand=m_dom, dom_rand_tumor=m_dt)
        with open("sfcn_eval_results.json", "w") as f:
            json.dump(out, f, indent=2)


        print("\n✓ All evaluations done – results saved.")
        return out

    except Exception as e:
        print(f"❌ Error during inference: {type(e).__name__}: {e}")
        raise
    finally:
        wandb.finish()


# ─────────────────────────── CLI entry point ────────────────────────────────
if __name__ == "__main__":
    inference_with_3fold_evaluation()