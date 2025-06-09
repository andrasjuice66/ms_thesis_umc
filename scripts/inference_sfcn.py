#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SFCN 3-fold evaluation / inference script
Fixed version – 2025-06-09
"""

# ───────────────────────────────── Imports ──────────────────────────────────
import io
import json
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
import pickle


# Project-local utilities
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer
from brain_age_pred.training.metrics import calculate_metrics
from brain_age_pred.utils.utils import read_csv


def safe_torch_load(fp, map_location="cpu"):
    """
    Load a checkpoint, transparently handling UTF-8 BOMs and the
    weights_only change in PyTorch ≥ 2.6.
    """
    def _try_load(handle):
        # First try the full-checkpoint route (works on ≥2.6, may raise
        # TypeError on <2.6 where weights_only is unknown)
        try:
            return torch.load(handle, map_location=map_location,
                              weights_only=False)
        except TypeError:          # < 2.6 fallback
            return torch.load(handle, map_location=map_location)

    try:
        # ① normal path
        return _try_load(fp)

    except (pickle.UnpicklingError, RuntimeError):
        # ② maybe the file starts with a UTF-8 BOM — strip & retry
        with open(fp, "rb") as f:
            raw = f.read()
        bom = b"\xef\xbb\xbf"
        if raw.startswith(bom):
            print("⚠️  UTF-8 BOM detected – stripping it and re-loading …")
            return _try_load(io.BytesIO(raw[len(bom):]))
        raise      # not a BOM issue → propagate


def num2vect(x, bin_range, bin_step, sigma):
    """
    Convert a number or array of numbers to a (soft) one-hot vector.
    """
    bin_start, bin_end = bin_range
    bin_length = bin_end - bin_start
    if bin_length % bin_step != 0:
        raise ValueError("bin_length must be divisible by bin_step")
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + bin_step * (0.5 + np.arange(bin_number))

    if sigma == 0:  # hard label
        idx = np.floor((np.asarray(x) - bin_start) / bin_step).astype(int)
        return idx, bin_centers

    # soft label
    x = np.asarray(x).reshape(-1)
    v = np.zeros((x.size, bin_number), dtype=np.float32)
    for j, age in enumerate(x):
        for i in range(bin_number):
            x1 = bin_centers[i] - bin_step / 2
            x2 = bin_centers[i] + bin_step / 2
            cdfs = norm.cdf([x1, x2], loc=age, scale=sigma)
            v[j, i] = cdfs[1] - cdfs[0]
    return v.squeeze(), bin_centers


def crop_center(data, out_sp):
    """
    Center-crop a 3-D (or 4-D with channel) volume to `out_sp`.
    """
    in_sp = data.shape
    if data.ndim not in (3, 4):
        raise ValueError(f"Wrong dimension! dim={data.ndim}.")
    # z, y, x (last three dims)
    dz = (in_sp[-3] - out_sp[-3]) // 2
    dy = (in_sp[-2] - out_sp[-2]) // 2
    dx = (in_sp[-1] - out_sp[-1]) // 2
    if data.ndim == 3:
        return data[dz:-dz, dy:-dy, dx:-dx]
    else:
        return data[:, dz:-dz, dy:-dy, dx:-dx]


def my_KLDivLoss(x, y):
    """
    Batch-wise KL-divergence (averaged over batch).
    """
    y = y + 1e-16
    loss = F.kl_div(x, y, reduction="sum") / y.size(0)
    return loss


# ───────────────────────────── Model Definition ─────────────────────────────
class SFCN(nn.Module):
    def __init__(self, channel_number=(32, 64, 128, 256, 256, 64),
                 output_dim=40, dropout=True):
        super().__init__()
        self.feature_extractor = nn.Sequential()
        for i, out_ch in enumerate(channel_number):
            in_ch = 1 if i == 0 else channel_number[i - 1]
            maxpool = i < len(channel_number) - 1
            k, p = (3, 1) if maxpool else (1, 0)
            self.feature_extractor.add_module(
                f"conv_{i}", self.conv_layer(in_ch, out_ch,
                                             maxpool=maxpool,
                                             kernel_size=k,
                                             padding=p)
            )

        self.classifier = nn.Sequential(
            nn.AvgPool3d((5, 6, 5)),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Conv3d(channel_number[-1], output_dim, 1)
        )

    @staticmethod
    def conv_layer(in_ch, out_ch, *, maxpool=True,
                   kernel_size=3, padding=0, maxpool_stride=2):
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if maxpool:
            layers.insert(2, nn.MaxPool3d(2, stride=maxpool_stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return [F.log_softmax(x, dim=1)]  # keep original list API


# ───────────────────────────── Data utilities ───────────────────────────────
def create_test_dataloader(csv_path, data_dir, transform=None,
                           batch_size=8, num_workers=4):
    file_paths, ages, sample_weights, sexes, modalities = read_csv(
        csv_path, data_dir
    )
    dataset = BADataset(
        file_paths=file_paths,
        age_labels=ages,
        sexes=sexes,
        modalities=modalities,
        transform=transform,
        mode="test",
        cache_size=0,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return loader, file_paths, ages, sexes, modalities


def create_eval_transforms(device, *, use_domain_rand=False, use_tumor=False):
    if not use_domain_rand:
        return None

    eval_rand_cfg = {
        "use_domain_randomization": True,
        "transform_probs": {
            "flip": 0.5, "affine": 0.8, "contrast": 0.6, "gamma": 0.5,
            "blur": 0.4, "bias": 0.5, "scale_int": 0.4, "shift_int": 0.4,
            "hist_shift": 0.3, "noise": 0.4, "rician": 0.3, "gibbs": 0.3,
            "resolution": 0.5, "coarse_do": 0.3, "crop": 1.0,
            "tumor": 0.3 if use_tumor else 0.0,
        },
        "output_shape": (160, 192, 160),
    }

    eval_tumor_cfg = {
        "use_tumor_simulation": use_tumor,
        "prob": 0.3,
        "use_age_based_segmentation": False,
        "perlin_res": [2, 2, 2],
        "tumor_size_factor_range": [0.5, 2.0],
        "use_fluid_dynamics": True,
    } if use_tumor else {}

    return DomainRandomizer(
        device=device,
        use_tumor_simulation=use_tumor,
        tumor_config=eval_tumor_cfg,
        **eval_rand_cfg,
    )


# ───────────────────────────── Evaluation loops ─────────────────────────────
def run_single_evaluation(model, loader, device,
                          bin_range, bin_step, sigma, bin_centers):
    model.eval()
    preds, targs, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            ages = batch["age"].to(device)

            soft = [
                num2vect(a.item(), bin_range, bin_step, sigma)[0]
                for a in ages
            ]
            soft = torch.as_tensor(soft, dtype=torch.float32, device=device)

            log_probs = model(imgs)[0]
            losses.append(my_KLDivLoss(log_probs, soft).item())

            probs = torch.exp(log_probs)
            pred = (probs * torch.tensor(bin_centers, device=device)).sum(1)
            preds.append(pred.cpu().numpy())
            targs.append(ages.cpu().numpy())

    return (np.concatenate(preds), np.concatenate(targs),
            float(np.mean(losses)))


def run_multi_fold_evaluation(model, csv_path, data_dir, device, transform,
                              n_folds, eval_name,
                              bin_range, bin_step, sigma, bin_centers,
                              batch_size=8):
    print(f"Running {n_folds}-fold {eval_name} evaluation …")
    fold_metrics = []

    for k in range(n_folds):
        print(f"{eval_name} fold {k+1}/{n_folds}")
        loader, _, _, sexes, modalities = create_test_dataloader(
            csv_path, data_dir, transform, batch_size
        )
        preds, targs, loss = run_single_evaluation(
            model, loader, device,
            bin_range, bin_step, sigma, bin_centers
        )
        m = calculate_metrics(preds, targs, modalities, sexes)
        m["loss"] = loss
        fold_metrics.append(m)

    # aggregate
    out = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        out[key] = np.mean(vals)
        out[f"{key}_std"] = np.std(vals)
    return out


# ────────────────────────── Main inference function ─────────────────────────
def inference_with_3fold_evaluation():
    # Paths & options – adjust if needed
    model_path = "/home/ajoos/model_files/sfcn_original_ckp.p"
    test_csv_path = "/home/ajoos/brain_age_pred/data/labels/test.csv"
    data_dir = "/scratch-shared/ajoos/"
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # W&B init
    wandb.login(key="2abdb867a9244072f2237704a3cacc77fa548dd8")
    exp_name = f"sfcn_original_{datetime.now():%Y%m%d_%H%M%S}"
    wandb.init(project="brainage-inference", name=exp_name,
               config=dict(model="SFCN", model_path=model_path,
                           test_csv=test_csv_path, batch_size=batch_size,
                           device=str(device), evaluation_type="3fold_inference",
                           bin_range=[42, 82], bin_step=1, sigma=1),
               reinit=True)

    print(f"Running on device: {device}")
    print(f"W&B run: {exp_name}")

    try:
        # ── Model loading ────────────────────────────────────────────────
        model = SFCN()
        model = torch.nn.DataParallel(model)
        print(f"Loading model from {model_path}")

        state = safe_torch_load(model_path, map_location="cpu")
        state_dict = state["state_dict"] if (
            isinstance(state, dict) and "state_dict" in state
        ) else state
        model.load_state_dict(state_dict)

        model.to(device).eval()
        print("✅ Model loaded successfully\n")

        # ── Params for age bins ─────────────────────────────────────────
        bin_range = [42, 82]
        bin_step = 1
        sigma = 1
        bin_centers = bin_range[0] + bin_step * (0.5 + np.arange(
            int((bin_range[1] - bin_range[0]) / bin_step))
        )

        # ── Dataset info logging ───────────────────────────────────────
        file_paths, ages, _, sexes, modalities = read_csv(
            test_csv_path, data_dir
        )
        wandb.log({
            "dataset/num_samples": len(file_paths),
            "dataset/age_min": float(np.min(ages)),
            "dataset/age_max": float(np.max(ages)),
            "dataset/age_mean": float(np.mean(ages)),
            "dataset/age_std": float(np.std(ages)),
        })

        # ── 1) Plain test evaluation ───────────────────────────────────
        print("=== 1/3: Plain test evaluation ===")
        loader, _, _, _, _ = create_test_dataloader(
            test_csv_path, data_dir, None, batch_size
        )
        preds, targs, loss = run_single_evaluation(
            model, loader, device,
            bin_range, bin_step, sigma, bin_centers
        )
        plain_metrics = calculate_metrics(preds, targs, modalities, sexes)
        plain_metrics["loss"] = loss
        print(f"Plain test MAE = {plain_metrics['mae']:.4f}, "
              f"R² = {plain_metrics['r2']:.4f}")
        wandb.log({f"test/{k}": v for k, v in plain_metrics.items()})

        # ── 2) Domain randomization ────────────────────────────────────
        print("=== 2/3: Domain randomized evaluation ===")
        dom_rand_tf = create_eval_transforms(device, use_domain_rand=True,
                                             use_tumor=False)
        dom_metrics = run_multi_fold_evaluation(
            model, test_csv_path, data_dir, device, dom_rand_tf, 10,
            "domain_randomized", bin_range, bin_step, sigma, bin_centers,
            batch_size
        )
        wandb.log({f"test_dom_rand/{k}": v for k, v in dom_metrics.items()})

        # ── 3) Domain rand + tumor ─────────────────────────────────────
        print("=== 3/3: Domain randomized + tumor evaluation ===")
        dom_tumor_tf = create_eval_transforms(device, use_domain_rand=True,
                                              use_tumor=True)
        dom_tumor_metrics = run_multi_fold_evaluation(
            model, test_csv_path, data_dir, device, dom_tumor_tf, 10,
            "domain_rand_tumor", bin_range, bin_step, sigma, bin_centers,
            batch_size
        )
        wandb.log({f"test_dom_rand_tumor/{k}": v
                   for k, v in dom_tumor_metrics.items()})

        # ── Save & visualise ───────────────────────────────────────────
        results = dict(plain=plain_metrics,
                       domain_rand=dom_metrics,
                       domain_rand_tumor=dom_tumor_metrics)
        with open("sfcn_3fold_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # scatter & bar plots
        plt.figure(figsize=(15, 5))

        # scatter
        plt.subplot(1, 3, 1)
        plt.scatter(targs, preds, alpha=0.5)
        lims = (min(targs), max(targs))
        plt.plot(lims, lims, "r--")
        plt.xlabel("True age");  plt.ylabel("Predicted age")
        plt.title(f"Plain test\nMAE={plain_metrics['mae']:.2f}, "
                  f"R²={plain_metrics['r2']:.2f}")
        plt.grid(True)

        # MAE bars
        plt.subplot(1, 3, 2)
        mae_vals = [plain_metrics['mae'],
                    dom_metrics['mae'], dom_tumor_metrics['mae']]
        mae_stds = [0, dom_metrics['mae_std'], dom_tumor_metrics['mae_std']]
        labels = ["Plain", "Domain-Rand", "Dom-Rand+Tumor"]
        plt.bar(labels, mae_vals, yerr=mae_stds, capsize=5)
        plt.ylabel("MAE");  plt.title("MAE comparison");  plt.grid(axis='y')

        # R² bars
        plt.subplot(1, 3, 3)
        r2_vals = [plain_metrics['r2'],
                   dom_metrics['r2'], dom_tumor_metrics['r2']]
        r2_stds = [0, dom_metrics['r2_std'], dom_tumor_metrics['r2_std']]
        plt.bar(labels, r2_vals, yerr=r2_stds, capsize=5)
        plt.ylabel("R²");  plt.title("R² comparison");  plt.grid(axis='y')

        plt.tight_layout()
        plt.savefig("sfcn_3fold_evaluation_comparison.png", dpi=300)
        wandb.log({"evaluation_plots":
                   wandb.Image("sfcn_3fold_evaluation_comparison.png")})
        plt.close()

        # Detailed CSV export
        pd.DataFrame(dict(
            file_path=file_paths,
            true_age=targs,
            predicted_age=preds,
            brain_age_delta=preds - targs,
            sex=sexes if sexes is not None else None,
            modality=modalities if modalities is not None else None,
        )).to_csv("sfcn_plain_test_results.csv", index=False)

        print("✓ All done. Results written to disk.")
        return results

    except Exception as e:
        print(f"❌ Error during inference: {type(e).__name__}: {e}")
        raise
    finally:
        wandb.finish()


# Backward-compatibility alias
def inference_with_dataloader():
    return inference_with_3fold_evaluation()


# ────────────────────────────────── CLI ─────────────────────────────────────
if __name__ == "__main__":
    inference_with_3fold_evaluation()