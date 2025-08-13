#!/usr/bin/env python
"""
Evaluation script to test trained models on various test sets.

This script takes a configuration file that specifies which models to evaluate
and which test CSVs to use. It loads each model from its checkpoint, runs
predictions on each specified test set, and computes evaluation metrics,
including overall MAE, MAE per modality, MAE per sex, and MAE per headmotion type.

The results are printed in a summary table for easy comparison and logged to W&B.
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from monai.transforms import Compose, ToTensord, EnsureChannelFirstd, SqueezeDimd, AsDiscreted
from brain_age_pred.dataset.custom_transformations import IntensityClipNormalizeD, ConvertLabelsD
from sklearn.linear_model import LinearRegression

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dataset.dataset import BADataset
from brain_age_pred.models.sfcn import SFCN
from brain_age_pred.models.sfcn_class import SFCNClass
from brain_age_pred.models.brainagenext import BrainAgeNeXt
from brain_age_pred.models.multi_head import MultiTaskBrainAge
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import read_csv, load_checkpoint
from brain_age_pred.brain_gen.labels import GENERATION_LABELS, GENERATION_CLASSES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

class TestGenerator:
    """
    A minimal test-time transform pipeline that applies intensity normalization.
    """
    def __init__(self, normalize=True):
        if normalize:
            self.transform = Compose([
                IntensityClipNormalizeD(
                    keys=["image"],
                    clip_percentiles=(1.0, 99.0),
                    normalise=True,
                    gamma_std=0.0,
                    prob=1.0
                )
            ])
        else:
            self.transform = Compose([])

    def __call__(self, data):
        return self.transform(data)

class SegmentationTestGenerator:
    """
    Test-time transform pipeline for segmentation models that require one-hot encoded input.
    """
    def __init__(self, normalize=True, n_classes=15):
        transforms = [
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ConvertLabelsD(
                keys=["image"],
                generation_labels=GENERATION_LABELS,
                output_labels=GENERATION_CLASSES
            ),
            SqueezeDimd(keys=["image"], dim=0),
            AsDiscreted(keys=["image"], to_onehot=n_classes),
        ]
        
        if normalize:
            transforms.append(
                IntensityClipNormalizeD(
                    keys=["image"],
                    clip_percentiles=(1.0, 99.0),
                    normalise=True,
                    gamma_std=0.0,
                    prob=1.0
                )
            )
        
        self.transform = Compose(transforms)

    def __call__(self, data):
        return self.transform(data)

def get_regression_correction_params(model, model_info, test_sets, global_data_dir, cfg, device, logger):
    """
    Calculates or loads regression correction parameters (alpha, beta) for a model.
    If a `regression_correction.json` file exists in the model's checkpoint directory, it loads from there.
    Otherwise, it calculates them using the 'OldValidationSet', saves them, and returns them.
    """
    model_type = model_info["params"].get("type", "").lower()
    if model_type not in ["brainagenext", "multitask"]:
        return None

    checkpoint_path = Path(model_info["checkpoint"])
    checkpoint_dir = checkpoint_path.parent
    correction_file = checkpoint_dir / "regression_correction.json"

    if correction_file.exists():
        logger.info(f"Loading regression correction parameters from {correction_file}")
        with open(correction_file, 'r') as f:
            return json.load(f)

    logger.info(f"Regression correction file not found. Calculating from 'OldValidationSet'...")
    val_set_info = next((item for item in test_sets if item["name"] == "OldValidationSet"), None)

    if not val_set_info:
        logger.warning("Could not find 'OldValidationSet' in config to calculate regression correction. Skipping correction for this model.")
        return None

    # Copied logic from main loop to run evaluation on one dataset
    val_csv = Path(val_set_info["csv_path"])
    data_dir = Path(model_info.get("data_dir", global_data_dir))
    paths, ages, _, sexes, modalities, headmotions = read_csv_with_headmotion(val_csv, data_dir)

    model_params = model_info["params"]
    in_channels = model_params.get("in_channels", 1)
    should_normalize = model_params.get("normalize", False)
    if in_channels > 1:
        transform = SegmentationTestGenerator(normalize=should_normalize, n_classes=in_channels)
    else:
        transform = TestGenerator(normalize=should_normalize)

    val_ds = BADataset(file_paths=paths, age_labels=ages, sexes=sexes, modalities=modalities, mode="test", transform=transform)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8), shuffle=False, num_workers=cfg.get("num_workers", 4))

    val_preds, val_true_ages, _, _, _ = evaluate_model(model, val_loader, device, model_type, headmotions=None)

    reg = LinearRegression()
    reg.fit(val_true_ages.reshape(-1, 1), val_preds)

    alpha = reg.coef_[0]
    beta = reg.intercept_

    correction_params = {'alpha': float(alpha), 'beta': float(beta)}
    logger.info(f"Calculated correction params: alpha={alpha:.4f}, beta={beta:.4f}")

    with open(correction_file, 'w') as f:
        json.dump(correction_params, f, indent=4)
    logger.info(f"Saved correction parameters to {correction_file}")

    return correction_params


def read_csv_with_headmotion(
    csv_path: str,
    data_root: str,
    image_key: str = "image_path",
    age_key: str = "age",
    weight_key: str = "sample_weight", 
    sex_key: str = "sex",
    modalities_key: str = "modality",
    headmotion_key: str = "headmotion",
    min_age: float = 20.0,
    max_age: float = 80.0,
):
    """Extended version of read_csv that also extracts headmotion data if available and filters by age."""
    df = pd.read_csv(csv_path)
    paths, ages, weights, sexes, modalities, headmotions = [], [], [], [], [], []
    data_root = Path(data_root)
    
    total_samples = 0
    filtered_samples = 0
    
    for _, row in df.iterrows():
        total_samples += 1
        age = float(row[age_key])
        
        # Skip samples outside age range
        if age < min_age or age > max_age:
            continue
            
        rel_path = row[image_key]
        fpath = data_root / rel_path
        if fpath.exists():
            filtered_samples += 1
            paths.append(str(fpath))
            ages.append(age)
            weights.append(float(row.get(weight_key, 1.0)))
            sexes.append(str(row.get(sex_key, 'N/A')))
            modalities.append(str(row.get(modalities_key, 'N/A')))
            headmotions.append(str(row.get(headmotion_key, 'N/A')))
    
    print(f"Age filtering: {filtered_samples}/{total_samples} samples kept (ages {min_age}-{max_age})")
    return paths, ages, weights, sexes, modalities, headmotions

def get_model(model_config, device):
    """Initializes a model based on the provided configuration."""
    mtype = model_config.get("type", "sfcn").lower()
    
    if mtype == "sfcn":
        model = SFCN(
            in_channels=model_config.get("in_channels"),
            dropout_rate=model_config.get("dropout_rate"),
            age_min=model_config.get("age_min"),
            age_max=model_config.get("age_max"),
        )
    elif mtype == "sfcn_class":
        model = SFCNClass(
            in_channels=model_config.get("in_channels"),
            dropout_rate=model_config.get("dropout_rate"),
            channels=model_config.get("channels", (32, 64, 128, 256, 256, 64)),
            age_min=model_config.get("age_min"),
            age_max=model_config.get("age_max"),
            track_running_stats=model_config.get("track_running_stats", True),
        )
    elif mtype == "brainagenext":
        model = BrainAgeNeXt(
            in_channels=model_config.get("in_channels"),
            dropout_rate=model_config.get("dropout_rate"),
            model_id=model_config.get("model_id", "B"),
            kernel_size=model_config.get("kernel_size", 3),
            deep_supervision=model_config.get("deep_supervision", True),
            feature_size=model_config.get("feature_size", 512),
            hidden_size=model_config.get("hidden_size", 64),
        )
    elif mtype == "multitask":
        model = MultiTaskBrainAge(
            n_classes=len(GENERATION_LABELS),
            encoder_chs=model_config.get("encoder_chs", (24, 48, 96, 192, 384)),
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")
        
    return model.to(device)

def evaluate_model(model, test_loader, device, model_type, headmotions=None):
    """Runs the evaluation loop for a given model and test loader."""
    model.eval()
    all_preds = []
    all_ages = []
    all_modalities = []
    all_sexes = []
    all_headmotions = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            ages = batch["age"].numpy()
            modalities = batch["modality"]
            sexes = batch["sex"]

            output = model(images)

            if model_type == "multitask":
                _, preds_tensor = output  # seg_logits, age_pred
                preds = preds_tensor.cpu().numpy()
            elif model_type == "sfcn_class":
                preds = model.expected_age(output).cpu().numpy()
            else:
                preds = output.cpu().numpy()

            all_preds.extend(preds)
            all_ages.extend(ages)
            all_modalities.extend(modalities)
            all_sexes.extend(sexes)
            
            # Add headmotion data if available
            if headmotions is not None:
                batch_size = len(ages)
                start_idx = i * test_loader.batch_size
                end_idx = start_idx + batch_size
                batch_headmotions = headmotions[start_idx:end_idx]
                all_headmotions.extend(batch_headmotions)
            
    return np.array(all_preds), np.array(all_ages), all_modalities, all_sexes, all_headmotions

def _bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {}
    indices = np.arange(n)

    def mae(a, b): return np.mean(np.abs(a - b))
    def mse(a, b): return np.mean((a - b) ** 2)
    def r2(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    def corr(a, b):
        return np.corrcoef(a, b)[0, 1] if len(a) > 1 else 0.0

    maes, mses, r2s, cors, bad_means = [], [], [], [], []
    for _ in range(n_boot):
        bs_idx = rng.choice(indices, size=n, replace=True)
        yt = y_true[bs_idx]
        yp = y_pred[bs_idx]
        maes.append(mae(yt, yp))
        mses.append(mse(yt, yp))
        r2s.append(r2(yt, yp))
        cors.append(corr(yt, yp))
        bad_means.append(np.mean(yp - yt))

    def ci(arr): 
        low, high = np.percentile(arr, [2.5, 97.5])
        return float(low), float(high)

    return {
        "mae": ci(maes),
        "mse": ci(mses),
        "r2": ci(r2s),
        "correlation": ci(cors),
        "bad_mean": ci(bad_means),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }

def calculate_metrics_with_ci(predictions, true_ages, modalities, sexes, headmotions=None, n_boot=1000, seed=42):
    """Calculates MAE, MSE, R², and correlation overall, per modality, per sex, and per headmotion type with confidence intervals."""
    df_data = {
        'prediction': predictions,
        'age': true_ages,
        'modality': modalities,
        'sex': sexes,
    }
    
    if headmotions:
        df_data['headmotion'] = headmotions
        
    df = pd.DataFrame(df_data)
    df['ae'] = np.abs(df['prediction'] - df['age'])
    df['se'] = (df['prediction'] - df['age']) ** 2
    df['bad'] = df['prediction'] - df['age']  # Brain Age Delta
    
    # Overall metrics - convert to native Python types
    mae = float(df['ae'].mean())
    mse = float(df['se'].mean())
    bad_mean = float(df['bad'].mean())  # Overall BAD mean
    
    # Calculate R² and correlation
    y_true = df['age'].values
    y_pred = df['prediction'].values
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    
    # Calculate overall bootstrap CIs
    overall_ci = _bootstrap_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
    
    metrics = {
        'overall_mae': mae,
        'overall_mse': mse,
        'overall_r2': r2,
        'overall_correlation': correlation,
        'overall_bad_mean': bad_mean,  # Add BAD mean
        'overall_bootstrap_ci': overall_ci,  # Add bootstrap CIs
        'count': len(df)
    }
    
    # MAE per modality with CIs
    modality_metrics = {}
    for modality in df['modality'].unique():
        mod_df = df[df['modality'] == modality]
        if len(mod_df) > 0:
            mod_mae = float(mod_df['ae'].mean())
            mod_mse = float(mod_df['se'].mean())
            mod_bad_mean = float(mod_df['bad'].mean())
            
            mod_y_true = mod_df['age'].values
            mod_y_pred = mod_df['prediction'].values
            
            mod_ss_res = np.sum((mod_y_true - mod_y_pred) ** 2)
            mod_ss_tot = np.sum((mod_y_true - np.mean(mod_y_true)) ** 2)
            mod_r2 = float(1 - (mod_ss_res / mod_ss_tot)) if mod_ss_tot != 0 else 0.0
            
            mod_correlation = float(np.corrcoef(mod_y_true, mod_y_pred)[0, 1]) if len(mod_y_true) > 1 else 0.0
            
            # Calculate modality-specific bootstrap CIs
            mod_ci = _bootstrap_ci(mod_y_true, mod_y_pred, n_boot=n_boot, seed=seed)
            
            modality_metrics[modality] = {
                'mae': mod_mae,
                'mse': mod_mse,
                'r2': mod_r2,
                'correlation': mod_correlation,
                'bad_mean': mod_bad_mean,
                'bootstrap_ci': mod_ci,
                'count': len(mod_df)
            }
    
    metrics['modality_metrics'] = modality_metrics
    
    # MAE per sex with CIs
    sex_metrics = {}
    for sex in df['sex'].unique():
        sex_df = df[df['sex'] == sex]
        if len(sex_df) > 0:
            sex_mae = float(sex_df['ae'].mean())
            sex_mse = float(sex_df['se'].mean())
            sex_bad_mean = float(sex_df['bad'].mean())
            
            sex_y_true = sex_df['age'].values
            sex_y_pred = sex_df['prediction'].values
            
            sex_ss_res = np.sum((sex_y_true - sex_y_pred) ** 2)
            sex_ss_tot = np.sum((sex_y_true - np.mean(sex_y_true)) ** 2)
            sex_r2 = float(1 - (sex_ss_res / sex_ss_tot)) if sex_ss_tot != 0 else 0.0
            
            sex_correlation = float(np.corrcoef(sex_y_true, sex_y_pred)[0, 1]) if len(sex_y_true) > 1 else 0.0
            
            # Calculate sex-specific bootstrap CIs
            sex_ci = _bootstrap_ci(sex_y_true, sex_y_pred, n_boot=n_boot, seed=seed)
            
            sex_metrics[sex] = {
                'mae': sex_mae,
                'mse': sex_mse,
                'r2': sex_r2,
                'correlation': sex_correlation,
                'bad_mean': sex_bad_mean,
                'bootstrap_ci': sex_ci,
                'count': len(sex_df)
            }
    
    metrics['sex_metrics'] = sex_metrics
    
    # MAE per headmotion type (if available) with CIs
    if headmotions and any(h != 'N/A' for h in headmotions):
        headmotion_metrics = {}
        # Map headmotion codes to readable names
        headmotion_mapping = {
            '0': 'Standard',
            '1': 'HeadMotion1', 
            '2': 'HeadMotion2',
            'N/A': 'N/A'
        }
        
        for headmotion in df['headmotion'].unique():
            hm_df = df[df['headmotion'] == headmotion]
            if len(hm_df) > 0:
                hm_mae = float(hm_df['ae'].mean())
                hm_mse = float(hm_df['se'].mean())
                hm_bad_mean = float(hm_df['bad'].mean())
                
                hm_y_true = hm_df['age'].values
                hm_y_pred = hm_df['prediction'].values
                
                hm_ss_res = np.sum((hm_y_true - hm_y_pred) ** 2)
                hm_ss_tot = np.sum((hm_y_true - np.mean(hm_y_true)) ** 2)
                hm_r2 = float(1 - (hm_ss_res / hm_ss_tot)) if hm_ss_tot != 0 else 0.0
                
                hm_correlation = float(np.corrcoef(hm_y_true, hm_y_pred)[0, 1]) if len(hm_y_true) > 1 else 0.0
                
                # Calculate headmotion-specific bootstrap CIs
                hm_ci = _bootstrap_ci(hm_y_true, hm_y_pred, n_boot=n_boot, seed=seed)
                
                readable_name = headmotion_mapping.get(str(headmotion), str(headmotion))
                headmotion_metrics[readable_name] = {
                    'mae': hm_mae,
                    'mse': hm_mse,
                    'r2': hm_r2,
                    'correlation': hm_correlation,
                    'bad_mean': hm_bad_mean,
                    'bootstrap_ci': hm_ci,
                    'count': len(hm_df)
                }
        
        metrics['headmotion_metrics'] = headmotion_metrics
    
    return metrics

def compute_uncertainty_and_tests(y_true, y_pred, n_boot=1000, seed=42):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    bad = y_pred - y_true

    # Bootstrap 95% CIs
    ci = _bootstrap_ci(y_true, y_pred, n_boot=n_boot, seed=seed)

    # Significance tests
    pearson_r, pearson_p = (np.nan, np.nan)
    if len(y_true) > 1:
        try:
            pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        except Exception:
            pass

    bad_t, bad_p = (np.nan, np.nan)
    if len(bad) > 0:
        try:
            bad_t, bad_p = stats.ttest_1samp(bad, popmean=0.0, alternative="two-sided")
        except Exception:
            pass

    return {
        "bootstrap_95ci": ci,  # dict with ('low','high') tuples per metric
        "tests": {
            "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else None,
            "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else None,
            "bad_mean": float(np.mean(bad)) if len(bad) else None,
            "bad_t": float(bad_t) if np.isfinite(bad_t) else None,
            "bad_p": float(bad_p) if np.isfinite(bad_p) else None,
        }
    }

def _sanitize_name(s: str) -> str:
    return str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")

def save_eval_plots(y_true, y_pred, out_dir: Path, model_name: str, test_set_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predicted vs true
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    mn, mx = float(np.min(y_true + y_pred) / 2), float(np.max(y_true + y_pred) / 2)
    mn = float(np.min([np.min(y_true), np.min(y_pred)]))
    mx = float(np.max([np.max(y_true), np.max(y_pred)]))
    plt.plot([mn, mx], [mn, mx], 'r--', lw=1)
    plt.xlabel("Chronological age")
    plt.ylabel("Predicted age")
    plt.title(f"{model_name} vs Age ({test_set_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pred_vs_age.png", dpi=200)
    plt.close()

    # BAD vs age
    bad = y_pred - y_true
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, bad, s=8, alpha=0.6)
    plt.axhline(0.0, color='r', linestyle='--', lw=1)
    plt.xlabel("Chronological age")
    plt.ylabel("Brain Age Delta (Pred - Age)")
    plt.title(f"BAD vs Age ({model_name} | {test_set_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bad_vs_age.png", dpi=200)
    plt.close()

def print_summary_table(model_name, test_set_name, metrics):
    """Prints a formatted summary table of the evaluation metrics with confidence intervals."""
    print("\n" + "="*80)
    print(f"Evaluation Summary: Model '{model_name}' on Test Set '{test_set_name}'")
    print("-"*80)
    
    # Overall metrics with CIs
    overall_ci = metrics.get('overall_bootstrap_ci', {})
    mae_ci = overall_ci.get('mae')
    bad_ci = overall_ci.get('bad_mean')
    
    print(f"  Overall MAE: {metrics['overall_mae']:.4f}")
    if mae_ci:
        print(f"    MAE 95% CI: [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]")
    
    print(f"  Overall MSE: {metrics['overall_mse']:.4f}")
    print(f"  Overall R²:  {metrics['overall_r2']:.4f}")
    print(f"  Overall Correlation: {metrics['overall_correlation']:.4f}")
    print(f"  Overall BAD Mean: {metrics['overall_bad_mean']:.4f}")
    if bad_ci:
        print(f"    BAD Mean 95% CI: [{bad_ci[0]:.4f}, {bad_ci[1]:.4f}]")
    
    print(f"  Sample Count: {metrics['count']}")
    
    if 'uncertainty' in metrics:
        tests = metrics['uncertainty'].get('tests', {})
        if tests.get('pearson_p') is not None:
            print(f"  Pearson r p-value: {tests['pearson_p']:.2e}")
        if tests.get('bad_p') is not None:
            print(f"  BAD mean vs 0 p-value: {tests['bad_p']:.2e}")

    print("\n  Metrics by Modality:")
    if metrics.get('modality_metrics'):
        for modality, mod_metrics in metrics['modality_metrics'].items():
            mod_ci = mod_metrics.get('bootstrap_ci', {})
            mod_mae_ci = mod_ci.get('mae', (None, None))
            mod_bad_ci = mod_ci.get('bad_mean', (None, None))
            
            print(f"    - {modality} (n={mod_metrics['count']}): MAE={mod_metrics['mae']:.4f}, "
                  f"MSE={mod_metrics['mse']:.4f}, R²={mod_metrics['r2']:.4f}, "
                  f"Corr={mod_metrics['correlation']:.4f}, BAD={mod_metrics['bad_mean']:.4f}")
            if mod_mae_ci:
                print(f"      MAE 95% CI: [{mod_mae_ci[0]:.4f}, {mod_mae_ci[1]:.4f}]")
            if mod_bad_ci:
                print(f"      BAD 95% CI: [{mod_bad_ci[0]:.4f}, {mod_bad_ci[1]:.4f}]")
    else:
        print("    No modality data available.")
        
    print("\n  Metrics by Sex:")
    if metrics.get('sex_metrics'):
        for sex, sex_metrics_data in metrics['sex_metrics'].items():
            sex_ci = sex_metrics_data.get('bootstrap_ci', {})
            sex_mae_ci = sex_ci.get('mae', (None, None))
            sex_bad_ci = sex_ci.get('bad_mean', (None, None))
            
            print(f"    - {sex} (n={sex_metrics_data['count']}): MAE={sex_metrics_data['mae']:.4f}, "
                  f"MSE={sex_metrics_data['mse']:.4f}, R²={sex_metrics_data['r2']:.4f}, "
                  f"Corr={sex_metrics_data['correlation']:.4f}, BAD={sex_metrics_data['bad_mean']:.4f}")
            if sex_mae_ci:
                print(f"      MAE 95% CI: [{sex_mae_ci[0]:.4f}, {sex_mae_ci[1]:.4f}]")
            if sex_bad_ci:
                print(f"      BAD 95% CI: [{sex_bad_ci[0]:.4f}, {sex_bad_ci[1]:.4f}]")
    else:
        print("    No sex data available.")
    
    # Add headmotion results if available
    if 'headmotion_metrics' in metrics:
        print("\n  Metrics by Head Motion Type:")
        for headmotion, hm_metrics in metrics['headmotion_metrics'].items():
            hm_ci = hm_metrics.get('bootstrap_ci', {})
            hm_mae_ci = hm_ci.get('mae', (None, None))
            hm_bad_ci = hm_ci.get('bad_mean', (None, None))
            
            print(f"    - {headmotion} (n={hm_metrics['count']}): MAE={hm_metrics['mae']:.4f}, "
                  f"MSE={hm_metrics['mse']:.4f}, R²={hm_metrics['r2']:.4f}, "
                  f"Corr={hm_metrics['correlation']:.4f}, BAD={hm_metrics['bad_mean']:.4f}")
            if hm_mae_ci:
                print(f"      MAE 95% CI: [{hm_mae_ci[0]:.4f}, {hm_mae_ci[1]:.4f}]")
            if hm_bad_ci:
                print(f"      BAD 95% CI: [{hm_bad_ci[0]:.4f}, {hm_bad_ci[1]:.4f}]")
        
    print("="*80 + "\n")

def create_wandb_summary_table(evaluation_results):
    """Create a comprehensive wandb table showing all evaluation results."""
    table_data = []
    
    for model_name, test_results in evaluation_results.items():
        for test_set_name, metrics in test_results.items():
            # Overall metrics row
            overall_ci = metrics.get('overall_bootstrap_ci', {})
            mae_ci = overall_ci.get('mae', (None, None))
            
            table_data.append([
                model_name,
                test_set_name,
                "Overall",
                "N/A",
                metrics['count'],
                f"{metrics['overall_mae']:.4f}",
                f"[{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]" if mae_ci[0] is not None else "N/A",
                f"{metrics['overall_mse']:.4f}",
                f"{metrics['overall_r2']:.4f}",
                f"{metrics['overall_correlation']:.4f}",
                f"{metrics['overall_bad_mean']:.4f}"
            ])
            
            # Modality-specific rows
            if metrics.get('modality_metrics'):
                for modality, mod_metrics in metrics['modality_metrics'].items():
                    mod_ci = mod_metrics.get('bootstrap_ci', {})
                    mod_mae_ci = mod_ci.get('mae', (None, None))
                    
                    table_data.append([
                        model_name,
                        test_set_name,
                        "Modality",
                        modality,
                        mod_metrics['count'],
                        f"{mod_metrics['mae']:.4f}",
                        f"[{mod_mae_ci[0]:.4f}, {mod_mae_ci[1]:.4f}]" if mod_mae_ci[0] is not None else "N/A",
                        f"{mod_metrics['mse']:.4f}",
                        f"{mod_metrics['r2']:.4f}",
                        f"{mod_metrics['correlation']:.4f}",
                        f"{mod_metrics['bad_mean']:.4f}"
                    ])
            
            # Sex-specific rows
            if metrics.get('sex_metrics'):
                for sex, sex_metrics_data in metrics['sex_metrics'].items():
                    sex_ci = sex_metrics_data.get('bootstrap_ci', {})
                    sex_mae_ci = sex_ci.get('mae', (None, None))
                    
                    table_data.append([
                        model_name,
                        test_set_name,
                        "Sex",
                        sex,
                        sex_metrics_data['count'],
                        f"{sex_metrics_data['mae']:.4f}",
                        f"[{sex_mae_ci[0]:.4f}, {sex_mae_ci[1]:.4f}]" if sex_mae_ci[0] is not None else "N/A",
                        f"{sex_metrics_data['mse']:.4f}",
                        f"{sex_metrics_data['r2']:.4f}",
                        f"{sex_metrics_data['correlation']:.4f}",
                        f"{sex_metrics_data['bad_mean']:.4f}"
                    ])
            
            # Headmotion-specific rows
            if metrics.get('headmotion_metrics'):
                for headmotion, hm_metrics in metrics['headmotion_metrics'].items():
                    hm_ci = hm_metrics.get('bootstrap_ci', {})
                    hm_mae_ci = hm_ci.get('mae', (None, None))
                    
                    table_data.append([
                        model_name,
                        test_set_name,
                        "HeadMotion",
                        headmotion,
                        hm_metrics['count'],
                        f"{hm_metrics['mae']:.4f}",
                        f"[{hm_mae_ci[0]:.4f}, {hm_mae_ci[1]:.4f}]" if hm_mae_ci[0] is not None else "N/A",
                        f"{hm_metrics['mse']:.4f}",
                        f"{hm_metrics['r2']:.4f}",
                        f"{hm_metrics['correlation']:.4f}",
                        f"{hm_metrics['bad_mean']:.4f}"
                    ])
    
    # Create wandb table
    table = wandb.Table(
        columns=[
            "Model", "Test Set", "Category", "Subcategory", "Count",
            "MAE", "MAE 95% CI", "MSE", "R²", "Correlation", "BAD Mean"
        ],
        data=table_data
    )
    
    return table

def print_final_summary(evaluation_results):
    """Print a comprehensive final summary of all evaluations."""
    print("\n" + "="*100)
    print("FINAL EVALUATION SUMMARY - ALL MODELS AND TEST SETS")
    print("="*100)
    
    # Create summary table
    summary_data = []
    for model_name, test_results in evaluation_results.items():
        for test_set_name, metrics in test_results.items():
            summary_data.append({
                'Model': model_name,
                'Test Set': test_set_name,
                'MAE': metrics['overall_mae'],
                'MSE': metrics['overall_mse'],
                'R²': metrics['overall_r2'],
                'Correlation': metrics['overall_correlation'],
                'BAD Mean': metrics['overall_bad_mean'],
                'Count': metrics['count']
            })
    
    # Print as formatted table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(f"\n{'Model':<20} {'Test Set':<20} {'MAE':<8} {'MSE':<8} {'R²':<8} {'Corr':<8} {'BAD':<8} {'Count':<8}")
        print("-" * 108)
        for _, row in df.iterrows():
            print(f"{row['Model']:<20} {row['Test Set']:<20} {row['MAE']:<8.4f} {row['MSE']:<8.4f} "
                  f"{row['R²']:<8.4f} {row['Correlation']:<8.4f} {row['BAD Mean']:<8.4f} {row['Count']:<8}")
        
        # Print best performers
        print(f"\n{'-'*50}")
        print("BEST PERFORMERS:")
        print(f"{'-'*50}")
        best_mae_idx = df['MAE'].idxmin()
        best_r2_idx = df['R²'].idxmax()
        best_corr_idx = df['Correlation'].idxmax()
        
        print(f"Best MAE: {df.loc[best_mae_idx, 'Model']} on {df.loc[best_mae_idx, 'Test Set']} "
              f"(MAE: {df.loc[best_mae_idx, 'MAE']:.4f})")
        print(f"Best R²:  {df.loc[best_r2_idx, 'Model']} on {df.loc[best_r2_idx, 'Test Set']} "
              f"(R²: {df.loc[best_r2_idx, 'R²']:.4f})")
        print(f"Best Corr: {df.loc[best_corr_idx, 'Model']} on {df.loc[best_corr_idx, 'Test Set']} "
              f"(Corr: {df.loc[best_corr_idx, 'Correlation']:.4f})")
    
    print("="*100 + "\n")

def main():
    # 1. --- Configuration ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/evaluate_config.yaml"
    if not Path(cfg_file).exists():
        print(f"Error: Config file not found at '{cfg_file}'")
        sys.exit(1)
    cfg = Config(cfg_file)

    # 2. --- Setup ---
    log_dir = Path(cfg.get("output_dir", "output/evaluation"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("evaluation", log_file=log_dir / "eval.log")
    
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 3. --- W&B Setup ---
    use_wandb = cfg.get("wandb.use_wandb", True)
    experiment_name = cfg.get("wandb.experiment_name")
    if not experiment_name:
        experiment_name = f'evaluation_{timestamp}'
    
    if use_wandb:
        logger.info("Setting up Weights & Biases...")
        WANDB_API = '2abdb867a9244072f2237704a3cacc77fa548dd8'
        wandb.login(key=WANDB_API)
        wandb.init(
            project=cfg.get("wandb.project", "brain-age-evaluation"),
            entity=cfg.get("wandb.entity"),
            name=experiment_name,
            config=cfg.config,
            reinit=True,
        )
        logger.info("W&B initialized successfully")

    # 4. --- Evaluation Loop ---
    evaluation_results = {}
    models_to_eval = cfg.get("models", [])
    test_sets = cfg.get("testing", [])
    global_data_dir = Path(cfg.get("data_dir"))

    if not models_to_eval or not test_sets:
        logger.error("Config file must contain 'models' and 'testing' sections.")
        return

    for model_info in models_to_eval:
        model_name = model_info["name"]
        logger.info(f"--- Evaluating model: {model_name} ---")
        evaluation_results[model_name] = {}
        
        # Initialize model
        model = get_model(model_info["params"], device)
        
        # Load checkpoint 
        checkpoint_path = model_info["checkpoint"]
        try:
            load_checkpoint(model, checkpoint_path, device, logger)
        except Exception as e:
            logger.error(f"Could not load checkpoint for model '{model_name}'. Skipping.")
            continue

        # Get regression correction parameters
        correction_params = get_regression_correction_params(
            model, model_info, test_sets, global_data_dir, cfg, device, logger
        )

        for test_info in test_sets:
            test_set_name = test_info["name"]
            test_csv = Path(test_info["csv_path"])
            logger.info(f"--- Running on test set: {test_set_name} ---")
            
            if not test_csv.exists():
                logger.error(f"Test CSV for '{test_set_name}' not found at '{test_csv}'. Skipping.")
                continue

            # Determine data directory (model-specific or global)
            data_dir = Path(model_info.get("data_dir", global_data_dir))
            logger.info(f"Using data directory: {data_dir}")

            # Load data with headmotion support
            paths, ages, _, sexes, modalities, headmotions = read_csv_with_headmotion(
                test_csv,
                data_dir,
            )

            # Check if this model requires segmentation input and select appropriate transform
            model_params = model_info["params"]
            in_channels = model_params.get("in_channels", 1)
            should_normalize = model_params.get("normalize", False)

            if in_channels > 1:
                # Model expects multi-channel segmentation input
                n_classes = in_channels  # Assume in_channels equals number of segmentation classes
                transform = SegmentationTestGenerator(normalize=should_normalize, n_classes=n_classes)
                logger.info(f"Using segmentation transforms for model '{model_name}' with {n_classes} classes. Normalization: {'ENABLED' if should_normalize else 'DISABLED'}.")
            else:
                # Model expects single-channel input
                transform = TestGenerator(normalize=should_normalize)
                logger.info(f"Using standard transforms for model '{model_name}'. Normalization: {'ENABLED' if should_normalize else 'DISABLED'}.")

            test_ds = BADataset(
                file_paths=paths,
                age_labels=ages,
                sexes=sexes,
                modalities=modalities,
                mode="test",
                transform=transform,
            )

            test_loader = DataLoader(
                test_ds,
                batch_size=cfg.get("batch_size", 8),
                shuffle=False,
                num_workers=cfg.get("num_workers", 4),
            )

            # Evaluate and calculate metrics
            preds, true_ages, mods, sxs, hmotions = evaluate_model(
                model, test_loader, device, model_info["params"]["type"], headmotions
            )
            
            # Apply correction if params are available
            if correction_params:
                logger.info(f"Applying regression correction to predictions for '{test_set_name}'")
                alpha = correction_params['alpha']
                beta = correction_params['beta']
                preds = (preds - beta) / alpha

            metrics = calculate_metrics_with_ci(
                preds, true_ages, mods, sxs, hmotions,
                n_boot=cfg.get("n_bootstrap", 1000),
                seed=cfg.get("bootstrap_seed", 42)
            )
            
            uncertainty = compute_uncertainty_and_tests(
                true_ages, preds,
                n_boot=cfg.get("n_bootstrap", 1000),
                seed=cfg.get("bootstrap_seed", 42),
            )
            metrics['uncertainty'] = uncertainty

            # save plots
            plots_subdir = log_dir / "plots" / _sanitize_name(model_name) / _sanitize_name(test_set_name)
            save_eval_plots(true_ages, preds, plots_subdir, model_name, test_set_name)

            evaluation_results[model_name][test_set_name] = metrics
            
            # Log to W&B
            if use_wandb:
                log_prefix = f"{model_name}_{test_set_name}"
                wandb.log({
                    f"{log_prefix}/overall_mae": metrics['overall_mae'],
                    f"{log_prefix}/overall_mse": metrics['overall_mse'], 
                    f"{log_prefix}/overall_r2": metrics['overall_r2'],
                    f"{log_prefix}/overall_correlation": metrics['overall_correlation'],
                    f"{log_prefix}/overall_bad_mean": metrics['overall_bad_mean'],
                    f"{log_prefix}/count": metrics['count']
                })
                
                # Log modality-specific metrics
                if metrics.get('modality_metrics'):
                    for modality, mod_metrics in metrics['modality_metrics'].items():
                        wandb.log({
                            f"{log_prefix}/{modality}_mae": mod_metrics['mae'],
                            f"{log_prefix}/{modality}_mse": mod_metrics['mse'],
                            f"{log_prefix}/{modality}_r2": mod_metrics['r2'],
                            f"{log_prefix}/{modality}_correlation": mod_metrics['correlation'],
                            f"{log_prefix}/{modality}_bad_mean": mod_metrics['bad_mean'],
                            f"{log_prefix}/{modality}_count": mod_metrics['count']
                        })
                
                # Log sex-specific metrics
                if metrics.get('sex_metrics'):
                    for sex, sex_metrics_data in metrics['sex_metrics'].items():
                        wandb.log({
                            f"{log_prefix}/{sex}_mae": sex_metrics_data['mae'],
                            f"{log_prefix}/{sex}_mse": sex_metrics_data['mse'],
                            f"{log_prefix}/{sex}_r2": sex_metrics_data['r2'],
                            f"{log_prefix}/{sex}_correlation": sex_metrics_data['correlation'],
                            f"{log_prefix}/{sex}_bad_mean": sex_metrics_data['bad_mean'],
                            f"{log_prefix}/{sex}_count": sex_metrics_data['count']
                        })
                
                if use_wandb:
                    wandb.log({
                        f"{log_prefix}/pred_vs_age": wandb.Image(str(plots_subdir / "pred_vs_age.png")),
                        f"{log_prefix}/bad_vs_age": wandb.Image(str(plots_subdir / "bad_vs_age.png")),
                    })

            # Print summary
            print_summary_table(model_name, test_set_name, metrics)

    # 5. --- Final Summary and W&B Table ---
    print_final_summary(evaluation_results)
    
    if use_wandb:
        # Create and log comprehensive table
        logger.info("Creating comprehensive results table for W&B...")
        summary_table = create_wandb_summary_table(evaluation_results)
        wandb.log({"evaluation_summary_table": summary_table})
        
        # Log overall summary metrics
        all_maes = []
        for model_results in evaluation_results.values():
            for test_metrics in model_results.values():
                all_maes.append(test_metrics['overall_mae'])
        
        if all_maes:
            wandb.log({
                "summary/average_mae_across_all": np.mean(all_maes),
                "summary/best_mae": np.min(all_maes),
                "summary/worst_mae": np.max(all_maes),
                "summary/num_evaluations": len(all_maes)
            })

    # 6. --- Save final results ---
    results_file = log_dir / "evaluation_summary.json"
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    logger.info(f"Full evaluation results saved to {results_file}")
    
    if use_wandb:
        wandb.finish()
        logger.info("W&B session finished")

if __name__ == "__main__":
    main() 
    