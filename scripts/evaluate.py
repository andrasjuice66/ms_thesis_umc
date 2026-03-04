#!/usr/bin/env python
"""
Evaluation script to test trained models on various test sets.

This script takes a configuration file that specifies which models to evaluate
and which test CSVs to use. It loads each model from its checkpoint, runs
predictions on each specified test set, and computes evaluation metrics,
including overall MAE, MAE per modality, MAE per sex, and MAE per headmotion type.

The results are printed in a summary table for easy comparison.
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import wandb
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.transforms import Compose, EnsureChannelFirstd, SqueezeDimd, AsDiscreted

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
from brain_age_pred.dataset.custom_transformations import ConvertLabelsD
from brain_age_pred.brain_gen.labels import GENERATION_LABELS, GENERATION_CLASSES
from brain_age_pred.utils.gradcam import (
    generate_gradcam_samples, plot_gradcam_samples,
    generate_average_gradcam, plot_average_gradcam,
)

class SegmentationTestGenerator:
    """Test-time transform for models that require one-hot encoded segmentation input."""
    def __init__(self, n_classes=15):
        self.transform = Compose([
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ConvertLabelsD(
                keys=["image"],
                generation_labels=GENERATION_LABELS,
                output_labels=GENERATION_CLASSES
            ),
            SqueezeDimd(keys=["image"], dim=0),
            AsDiscreted(keys=["image"], to_onehot=n_classes),
        ])

    def __call__(self, data):
        return self.transform(data)

def read_csv_with_headmotion(
    csv_path: str,
    data_root: str,
    image_key: str = "image_path",
    age_key: str = "age",
    weight_key: str = "sample_weight", 
    sex_key: str = "sex",
    modalities_key: str = "modality",
    headmotion_key: str = "headmotion",
):
    """Extended version of read_csv that also extracts headmotion data if available."""
    df = pd.read_csv(csv_path)
    paths, ages, weights, sexes, modalities, headmotions = [], [], [], [], [], []
    data_root = Path(data_root)
    
    for _, row in df.iterrows():
        rel_path = row[image_key]
        fpath = data_root / rel_path
        if fpath.exists():
            paths.append(str(fpath))
            ages.append(float(row[age_key]))
            weights.append(float(row.get(weight_key, 1.0)))
            sexes.append(str(row.get(sex_key, 'N/A')))
            modalities.append(str(row.get(modalities_key, 'N/A')))
            headmotions.append(str(row.get(headmotion_key, 'N/A')))
    
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
            n_classes=model_config.get("n_classes"),
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
        for i, batch in enumerate(tqdm(test_loader, desc="  Evaluating", unit="batch", leave=True)):
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

            # Ensure preds/ages are at least 1-d (squeeze can produce 0-d for batch_size=1)
            preds = np.atleast_1d(preds)
            ages = np.atleast_1d(ages)

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

def fit_bias_correction(predictions: np.ndarray, true_ages: np.ndarray):
    """
    Fits the linear model x = a*y + b (predicted age = a * chronological age + b)
    on a labelled set (e.g. validation set) following Smith et al. (2019).

    Returns (a, b) so that the corrected predicted age is: x_corrected = (x - b) / a
    """
    a, b = np.polyfit(true_ages, predictions, 1)
    return float(a), float(b)


def apply_bias_correction(predictions: np.ndarray, a: float, b: float) -> np.ndarray:
    """Applies bias correction: x_corrected = (x - b) / a  (Smith et al., 2019)."""
    return (predictions - b) / a


def calculate_metrics(predictions, true_ages, modalities, sexes, headmotions=None):
    """Calculates MAE overall, per modality, per sex, and per headmotion type."""
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
    df['delta'] = df['prediction'] - df['age']

    spearman_r, spearman_p = spearmanr(df['delta'], df['age'])
    metrics = {
        'overall_mae': df['ae'].mean(),
        'spearman_r_delta_age': float(spearman_r),
        'spearman_p_delta_age': float(spearman_p),
    }
    
    # MAE per modality
    modality_mae = df.groupby('modality')['ae'].mean().to_dict()
    metrics['modality_mae'] = {m: v for m, v in modality_mae.items()}
    
    # MAE per sex
    sex_mae = df.groupby('sex')['ae'].mean().to_dict()
    metrics['sex_mae'] = {s: v for s, v in sex_mae.items()}
    
    # MAE per headmotion type (if available)
    if headmotions and any(h != 'N/A' for h in headmotions):
        headmotion_mae = df.groupby('headmotion')['ae'].mean().to_dict()
        # Map headmotion codes to readable names
        headmotion_mapping = {
            '0': 'Standard',
            '1': 'HeadMotion1', 
            '2': 'HeadMotion2',
            'N/A': 'N/A'
        }
        metrics['headmotion_mae'] = {
            headmotion_mapping.get(str(h), str(h)): v 
            for h, v in headmotion_mae.items()
        }
    
    return metrics

def print_summary_table(model_name, test_set_name, metrics, corrected_metrics=None, correction_coeffs=None):
    """Prints a formatted summary table of the evaluation metrics.

    If corrected_metrics is provided, shows before/after bias correction side-by-side.
    """
    print("\n" + "="*80)
    print(f"Evaluation Summary: Model '{model_name}' on Test Set '{test_set_name}'")
    if correction_coeffs is not None:
        a, b, restrict_mod = correction_coeffs
        scope = f"'{restrict_mod}' only" if restrict_mod else "all modalities"
        print(f"  Bias correction ({scope}): a={a:.4f}, b={b:.4f}  →  x_corrected = (x - b) / a")
    print("-"*80)

    if corrected_metrics is not None:
        print(f"  {'Metric':<30} {'Raw':>12} {'Corrected':>12}")
        print(f"  {'-'*30} {'-'*12} {'-'*12}")
        print(f"  {'Overall MAE':<30} {metrics['overall_mae']:>12.4f} {corrected_metrics['overall_mae']:>12.4f}")
        print(f"  {'Spearman r (delta vs age)':<30} {metrics['spearman_r_delta_age']:>12.4f} {corrected_metrics['spearman_r_delta_age']:>12.4f}")
    else:
        print(f"  Overall MAE: {metrics['overall_mae']:.4f}")
        print(f"  Spearman r (delta vs age): {metrics['spearman_r_delta_age']:.4f}  (p={metrics['spearman_p_delta_age']:.3e})")

    print("\n  MAE by Modality:")
    if metrics['modality_mae']:
        for modality, mae in metrics['modality_mae'].items():
            if corrected_metrics is not None:
                corr_mae = corrected_metrics['modality_mae'].get(modality, float('nan'))
                print(f"    - {modality:<26} raw={mae:.4f}  corrected={corr_mae:.4f}")
            else:
                print(f"    - {modality}: {mae:.4f}")
    else:
        print("    No modality data available.")

    print("\n  MAE by Sex:")
    if metrics['sex_mae']:
        for sex, mae in metrics['sex_mae'].items():
            if corrected_metrics is not None:
                corr_mae = corrected_metrics['sex_mae'].get(sex, float('nan'))
                print(f"    - {sex:<28} raw={mae:.4f}  corrected={corr_mae:.4f}")
            else:
                print(f"    - {sex}: {mae:.4f}")
    else:
        print("    No sex data available.")

    if 'headmotion_mae' in metrics:
        print("\n  MAE by Head Motion Type:")
        for headmotion, mae in metrics['headmotion_mae'].items():
            if corrected_metrics is not None:
                corr_mae = corrected_metrics.get('headmotion_mae', {}).get(headmotion, float('nan'))
                print(f"    - {headmotion:<26} raw={mae:.4f}  corrected={corr_mae:.4f}")
            else:
                print(f"    - {headmotion}: {mae:.4f}")

    print("="*80 + "\n")

def _safe_filename(s: str) -> str:
    """Strips characters that are unsafe in filenames."""
    for ch in (' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        s = s.replace(ch, '_')
    return s


def _try_set_style():
    for style in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot'):
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def plot_scatter_by_modality(
    model_name: str,
    test_set_name: str,
    preds: np.ndarray,
    true_ages: np.ndarray,
    modalities: list,
    output_dir: Path,
    corrected_preds: np.ndarray = None,
    use_wandb: bool = False,
) -> Path:
    """
    Scatter plots of chronological age vs predicted age, one subplot per modality
    plus an 'All' overview.  When bias-corrected predictions are supplied a second
    row (Corrected) is added beneath the Raw row.

    Returns the saved figure path.
    """
    _try_set_style()
    modalities_arr = np.array(modalities)
    unique_mods = sorted(set(modalities_arr))
    n_mods = len(unique_mods)
    cmap = matplotlib.colormaps.get_cmap('tab10').resampled(max(n_mods, 1))
    mod_color = {m: cmap(i) for i, m in enumerate(unique_mods)}

    n_rows = 2 if corrected_preds is not None else 1
    n_cols = n_mods + 1  # one per modality + "All" overview
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Model: {model_name}   |   Test set: {test_set_name}\nChronological Age vs Predicted Age",
        fontsize=12, fontweight='bold',
    )

    all_vals = np.concatenate([true_ages, preds] + ([corrected_preds] if corrected_preds is not None else []))
    pad = (all_vals.max() - all_vals.min()) * 0.05
    lim_lo, lim_hi = all_vals.min() - pad, all_vals.max() + pad

    row_specs = [("Raw", preds)]
    if corrected_preds is not None:
        row_specs.append(("Corrected", corrected_preds))

    for row_idx, (row_label, pred_set) in enumerate(row_specs):
        col_specs = [(mod, modalities_arr == mod) for mod in unique_mods]
        col_specs.append(("All", np.ones(len(true_ages), dtype=bool)))

        for col_idx, (mod_label, mask) in enumerate(col_specs):
            ax = axes[row_idx][col_idx]
            x, y = true_ages[mask], pred_set[mask]

            if mod_label == "All":
                for umod in unique_mods:
                    umask = modalities_arr == umod
                    ax.scatter(
                        true_ages[umask], pred_set[umask],
                        color=mod_color[umod], alpha=0.45, s=12,
                        label=umod, rasterized=True,
                    )
                if n_mods <= 10:
                    ax.legend(fontsize=6, markerscale=1.5, loc='upper left')
            else:
                ax.scatter(x, y, color=mod_color[mod_label], alpha=0.5, s=12, rasterized=True)

            # Identity line
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=1.2, alpha=0.6, label='y = x')
            # OLS regression line
            if len(x) >= 2:
                a_fit, b_fit = np.polyfit(x, y, 1)
                xs = np.array([lim_lo, lim_hi])
                ax.plot(xs, a_fit * xs + b_fit, color='crimson', lw=1.5,
                        label=f'fit: {a_fit:.2f}x + {b_fit:.1f}')

            mae_val = np.mean(np.abs(y - x))
            r_val, _ = spearmanr(y - x, x)
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("Chronological Age (yrs)", fontsize=8)
            ax.set_ylabel(f"{row_label}\nPredicted Age (yrs)" if col_idx == 0 else "Predicted Age (yrs)", fontsize=8)
            ax.set_title(
                f"{mod_label}\nMAE = {mae_val:.2f} yr   r = {r_val:.3f}",
                fontsize=9,
            )
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"scatter_{_safe_filename(model_name)}_{_safe_filename(test_set_name)}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if use_wandb:
        wandb.log({f"plots/scatter/{model_name}/{test_set_name}": wandb.Image(str(fname))})

    return fname


def plot_modality_bar_charts(
    evaluation_results: dict,
    output_dir: Path,
    use_wandb: bool = False,
) -> list:
    """
    Grouped bar charts comparing per-modality MAE across all models for each test set.

    One figure per test set.  When bias-corrected metrics are present, Raw and
    Corrected are shown in two vertically stacked panels within the same figure.
    Models are the grouped bars; modalities are the x-axis categories.

    Returns a list of saved figure paths.
    """
    _try_set_style()
    model_names = list(evaluation_results.keys())
    n_models = len(model_names)
    cmap = matplotlib.colormaps.get_cmap('Set2').resampled(max(n_models, 1))
    model_color = {m: cmap(i) for i, m in enumerate(model_names)}

    all_test_sets = sorted({ts for res in evaluation_results.values() for ts in res})
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for test_set_name in all_test_sets:
        all_modalities: set = set()
        for m_name in model_names:
            mets = evaluation_results[m_name].get(test_set_name, {})
            all_modalities.update(mets.get('modality_mae', {}).keys())
        modalities = sorted(all_modalities)
        if not modalities:
            continue

        has_corrected = any(
            evaluation_results[m].get(test_set_name, {}).get('corrected') is not None
            for m in model_names
        )
        n_rows = 2 if has_corrected else 1
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(max(9, 2.2 * len(modalities) * n_models), 5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Per-Modality MAE Comparison   |   Test set: {test_set_name}",
            fontsize=13, fontweight='bold',
        )

        x = np.arange(len(modalities))
        bar_w = 0.8 / n_models

        row_specs = [("Raw", 'modality_mae')]
        if has_corrected:
            row_specs.append(("Corrected", '_corrected_modality_mae'))

        for row_idx, (row_label, _) in enumerate(row_specs):
            ax = axes[row_idx][0]
            for m_idx, m_name in enumerate(model_names):
                mets = evaluation_results[m_name].get(test_set_name, {})
                if row_label == "Corrected":
                    mod_mae = mets.get('corrected', {}).get('modality_mae', {})
                else:
                    mod_mae = mets.get('modality_mae', {})
                values = [mod_mae.get(mod, float('nan')) for mod in modalities]
                offsets = x + (m_idx - (n_models - 1) / 2) * bar_w
                bars = ax.bar(
                    offsets, values, bar_w,
                    label=m_name, color=model_color[m_name], alpha=0.85, edgecolor='white',
                )
                for bar, val in zip(bars, values):
                    if not np.isnan(val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            f'{val:.2f}',
                            ha='center', va='bottom', fontsize=7, rotation=40,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(modalities, fontsize=10)
            ax.set_ylabel("MAE (years)", fontsize=11)
            ax.set_title(f"{row_label} Predictions", fontsize=11)
            ax.legend(fontsize=9, loc='upper right')
            valid_vals = [v for m in model_names
                          for v in ([evaluation_results[m].get(test_set_name, {}).get('corrected', {}).get('modality_mae', {}).get(mod, float('nan'))
                                      if row_label == "Corrected"
                                      else evaluation_results[m].get(test_set_name, {}).get('modality_mae', {}).get(mod, float('nan'))
                                      for mod in modalities])
                          if not np.isnan(v)]
            if valid_vals:
                ax.set_ylim(0, max(valid_vals) * 1.25)
            ax.grid(axis='y', alpha=0.4)

        plt.tight_layout()
        fname = output_dir / f"bar_modality_{_safe_filename(test_set_name)}.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(fname)

        if use_wandb:
            wandb.log({f"plots/bar_modality/{test_set_name}": wandb.Image(str(fname))})

    return saved_paths


def main():
    # 1. --- Configuration ---
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/evaluate_config.yaml"
    if not Path(cfg_file).exists():
        print(f"Error: Config file not found at '{cfg_file}'")
        sys.exit(1)
    cfg = Config(cfg_file)

    # 2. --- Setup ---
    log_dir = Path(cfg.get("output_dir", "output/evaluation"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("evaluation", log_file=log_dir / "eval.log")

    _plots_dir_raw = cfg.get("plots_dir")
    plots_dir = Path(_plots_dir_raw) if _plots_dir_raw else log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots will be saved to: {plots_dir}")
    
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 3. --- W&B Setup ---
    use_wandb = cfg.get("wandb.use_wandb", False)
    if use_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = cfg.get("wandb.experiment_name") or f"evaluation_{timestamp}"
        WANDB_API = '2abdb867a9244072f2237704a3cacc77fa548dd8'
        wandb.login(key=WANDB_API)
        wandb.init(
            project=cfg.get("wandb.project", "brain-age-evaluation"),
            entity=cfg.get("wandb.entity"),
            name=experiment_name,
            config=cfg.config,
            reinit=True,
        )
        logger.info(f"W&B initialized: project='{cfg.get('wandb.project')}', run='{experiment_name}'")

    # 4. --- Evaluation Loop ---
    evaluation_results = {}
    models_to_eval = cfg.get("models", [])
    test_sets = cfg.get("testing", [])
    global_data_dir = Path(cfg.get("data_dir"))

    # Bias correction settings (Smith et al., 2019)
    bc_enabled = cfg.get("bias_correction.enabled", False)
    bc_fit_on  = cfg.get("bias_correction.fit_on", "ValidationSet")

    # GradCAM settings
    gradcam_enabled   = cfg.get("gradcam.enabled", False)
    n_gradcam_samples     = cfg.get("gradcam.n_samples_per_modality", 2)
    n_gradcam_avg_samples = cfg.get("gradcam.n_avg_samples_per_modality", 50)

    if not models_to_eval or not test_sets:
        logger.error("Config file must contain 'models' and 'testing' sections.")
        return

    total_evals = len(models_to_eval) * len(test_sets)
    eval_count = 0
    for model_idx, model_info in enumerate(models_to_eval, 1):
        model_name = model_info["name"]
        logger.info(f"--- Evaluating model: {model_name} [{model_idx}/{len(models_to_eval)}] ---")
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

        # --- First pass: collect raw predictions for every test set ---
        raw_predictions = {}  # test_set_name -> (preds, true_ages, mods, sxs, hmotions)

        for test_idx, test_info in enumerate(test_sets, 1):
            test_set_name = test_info["name"]
            test_csv = Path(test_info["csv_path"])
            eval_count += 1
            logger.info(f"--- Running on test set: {test_set_name} [{test_idx}/{len(test_sets)}] (eval {eval_count}/{total_evals}) ---")

            if not test_csv.exists():
                logger.error(f"Test CSV for '{test_set_name}' not found at '{test_csv}'. Skipping.")
                continue

            # Determine data directory (model-specific or global)
            data_dir = Path(model_info.get("data_dir", global_data_dir))
            logger.info(f"Using data directory: {data_dir}")

            paths, ages, _, sexes, modalities, headmotions = read_csv_with_headmotion(
                test_csv,
                data_dir,
            )

            model_params = model_info["params"]
            in_channels = model_params.get("in_channels", 1)
            should_normalize = model_params.get("normalize", True)

            if in_channels > 1:
                transform = SegmentationTestGenerator(n_classes=in_channels)
                logger.info(f"Using segmentation transforms with {in_channels} classes.")
                test_ds = BADataset(
                    file_paths=paths,
                    age_labels=ages,
                    sexes=sexes,
                    modalities=modalities,
                    mode="test",
                    transform=transform,
                    normalize=False,
                    clamp=False,
                )
            else:
                test_ds = BADataset(
                    file_paths=paths,
                    age_labels=ages,
                    sexes=sexes,
                    modalities=modalities,
                    mode="test",
                    normalize=should_normalize,
                )

            test_loader = DataLoader(
                test_ds,
                batch_size=cfg.get("batch_size", 8),
                shuffle=False,
                num_workers=cfg.get("num_workers", 4),
            )

            preds, true_ages_arr, mods, sxs, hmotions = evaluate_model(
                model, test_loader, device, model_info["params"]["type"], headmotions
            )
            raw_predictions[test_set_name] = (preds, true_ages_arr, mods, sxs, hmotions)

            # GradCAM: generate heatmaps for a small representative sample per modality
            if gradcam_enabled:
                logger.info(f"Generating GradCAM for '{model_name}' / '{test_set_name}' "
                            f"({n_gradcam_samples} sample(s) per modality)...")
                gc_samples = generate_gradcam_samples(
                    model=model,
                    model_type=model_info["params"]["type"],
                    test_ds=test_ds,
                    modalities_list=modalities,
                    n_per_modality=n_gradcam_samples,
                    device=device,
                    log=logger,
                )
                if gc_samples:
                    gc_figs = plot_gradcam_samples(
                        model_name=model_name,
                        test_set_name=test_set_name,
                        samples=gc_samples,
                        output_dir=plots_dir,
                        use_wandb=use_wandb,
                    )
                    logger.info(f"GradCAM: saved {len(gc_figs)} individual figure(s) to "
                                f"{plots_dir / 'gradcam'}")
                else:
                    logger.warning("GradCAM: no individual samples were generated.")

                # Average GradCAM across all subjects per modality
                logger.info(f"Generating average GradCAM for '{model_name}' / "
                            f"'{test_set_name}' (up to {n_gradcam_avg_samples} "
                            f"subject(s) per modality)...")
                avg_results = generate_average_gradcam(
                    model=model,
                    model_type=model_info["params"]["type"],
                    test_ds=test_ds,
                    modalities_list=modalities,
                    n_max_per_modality=n_gradcam_avg_samples,
                    device=device,
                    log=logger,
                )
                if avg_results:
                    avg_fig = plot_average_gradcam(
                        model_name=model_name,
                        test_set_name=test_set_name,
                        avg_results=avg_results,
                        output_dir=plots_dir,
                        use_wandb=use_wandb,
                    )
                    if avg_fig:
                        logger.info(f"Average GradCAM saved: {avg_fig}")
                else:
                    logger.warning("Average GradCAM: no results generated.")

        # --- Bias correction: fit on the designated validation set ---
        # If the model declares `bias_correction_modality`, fitting and application
        # are restricted to that modality only (e.g. "t1" for a T1-only model).
        # All other subjects keep their raw predictions unchanged.
        correction_coeffs = None
        restrict_mod = model_info.get("bias_correction_modality")  # None or e.g. "t1"
        if bc_enabled:
            if bc_fit_on in raw_predictions:
                val_preds, val_ages, val_mods, _, _ = raw_predictions[bc_fit_on]
                if restrict_mod:
                    fit_mask = np.array(list(val_mods)) == restrict_mod
                    if fit_mask.sum() < 2:
                        logger.warning(
                            f"Bias correction: fewer than 2 '{restrict_mod}' subjects "
                            f"in '{bc_fit_on}'. Skipping correction for '{model_name}'."
                        )
                    else:
                        a, b = fit_bias_correction(val_preds[fit_mask], val_ages[fit_mask])
                        correction_coeffs = (a, b, restrict_mod)
                        logger.info(
                            f"Bias correction fitted on '{restrict_mod}' subjects only "
                            f"(n={fit_mask.sum()}) from '{bc_fit_on}': a={a:.4f}, b={b:.4f}"
                        )
                else:
                    a, b = fit_bias_correction(val_preds, val_ages)
                    correction_coeffs = (a, b, None)
                    logger.info(
                        f"Bias correction fitted on all subjects in '{bc_fit_on}': "
                        f"a={a:.4f}, b={b:.4f}"
                    )

                if correction_coeffs is not None:
                    a, b, _ = correction_coeffs
                    evaluation_results[model_name]['_bias_correction'] = {
                        'a': a, 'b': b,
                        'fitted_on': bc_fit_on,
                        'restricted_to_modality': restrict_mod,
                    }
                    if use_wandb:
                        wandb.log({
                            f"{model_name}/bias_correction/a": a,
                            f"{model_name}/bias_correction/b": b,
                            f"{model_name}/bias_correction/fitted_on": bc_fit_on,
                            f"{model_name}/bias_correction/restricted_to_modality":
                                restrict_mod or "all",
                        })
            else:
                logger.warning(
                    f"Bias correction enabled but '{bc_fit_on}' was not evaluated. "
                    "Correction will not be applied."
                )

        # --- Second pass: compute metrics (with and without correction) ---
        for test_set_name, (preds, true_ages_arr, mods, sxs, hmotions) in raw_predictions.items():
            metrics = calculate_metrics(preds, true_ages_arr, mods, sxs, hmotions)

            corrected_metrics = None
            corrected_preds = None
            if correction_coeffs is not None:
                a, b, restrict_mod = correction_coeffs
                if restrict_mod:
                    # Only correct subjects of the specified modality; leave others raw
                    corrected_preds = preds.astype(float).copy()
                    mod_mask = np.array(list(mods)) == restrict_mod
                    corrected_preds[mod_mask] = apply_bias_correction(preds[mod_mask], a, b)
                else:
                    corrected_preds = apply_bias_correction(preds, a, b)
                corrected_metrics = calculate_metrics(corrected_preds, true_ages_arr, mods, sxs, hmotions)
                metrics['corrected'] = corrected_metrics

            evaluation_results[model_name][test_set_name] = metrics

            print_summary_table(
                model_name, test_set_name, metrics,
                corrected_metrics=corrected_metrics,
                correction_coeffs=correction_coeffs,
            )

            # Scatter plot: chronological vs predicted age per modality
            scatter_path = plot_scatter_by_modality(
                model_name=model_name,
                test_set_name=test_set_name,
                preds=preds,
                true_ages=true_ages_arr,
                modalities=list(mods),
                output_dir=plots_dir,
                corrected_preds=corrected_preds,
                use_wandb=use_wandb,
            )
            logger.info(f"Scatter plot saved: {scatter_path}")

            # Log to W&B
            if use_wandb:
                log_prefix = f"{model_name}/{test_set_name}"
                wandb_log = {
                    f"{log_prefix}/raw/overall_mae": metrics['overall_mae'],
                    f"{log_prefix}/raw/spearman_r_delta_age": metrics['spearman_r_delta_age'],
                }
                for mod, mae in metrics.get('modality_mae', {}).items():
                    wandb_log[f"{log_prefix}/raw/modality/{mod}_mae"] = mae
                for sex, mae in metrics.get('sex_mae', {}).items():
                    wandb_log[f"{log_prefix}/raw/sex/{sex}_mae"] = mae
                for hm, mae in metrics.get('headmotion_mae', {}).items():
                    wandb_log[f"{log_prefix}/raw/headmotion/{hm}_mae"] = mae

                if corrected_metrics is not None:
                    wandb_log[f"{log_prefix}/corrected/overall_mae"] = corrected_metrics['overall_mae']
                    wandb_log[f"{log_prefix}/corrected/spearman_r_delta_age"] = corrected_metrics['spearman_r_delta_age']
                    for mod, mae in corrected_metrics.get('modality_mae', {}).items():
                        wandb_log[f"{log_prefix}/corrected/modality/{mod}_mae"] = mae
                    for sex, mae in corrected_metrics.get('sex_mae', {}).items():
                        wandb_log[f"{log_prefix}/corrected/sex/{sex}_mae"] = mae
                    for hm, mae in corrected_metrics.get('headmotion_mae', {}).items():
                        wandb_log[f"{log_prefix}/corrected/headmotion/{hm}_mae"] = mae

                wandb.log(wandb_log)

    # 5. --- Cross-model bar charts ---
    bar_paths = plot_modality_bar_charts(evaluation_results, plots_dir, use_wandb=use_wandb)
    for p in bar_paths:
        logger.info(f"Bar chart saved: {p}")

    # 6. --- W&B Summary Table ---
    if use_wandb:
        columns = [
            "Model", "Test Set",
            "MAE (raw)", "Spearman r (raw)",
            "MAE (corrected)", "Spearman r (corrected)",
            "Modality MAEs (raw)", "Sex MAEs (raw)",
        ]
        table = wandb.Table(columns=columns)
        for m_name, test_results in evaluation_results.items():
            for ts_name, mets in test_results.items():
                mod_str = ", ".join(f"{k}: {v:.4f}" for k, v in mets.get('modality_mae', {}).items())
                sex_str = ", ".join(f"{k}: {v:.4f}" for k, v in mets.get('sex_mae', {}).items())
                corr = mets.get('corrected')
                corr_mae = f"{corr['overall_mae']:.4f}" if corr else "N/A"
                corr_r = f"{corr['spearman_r_delta_age']:.4f}" if corr else "N/A"
                table.add_data(
                    m_name, ts_name,
                    f"{mets['overall_mae']:.4f}", f"{mets['spearman_r_delta_age']:.4f}",
                    corr_mae, corr_r,
                    mod_str, sex_str,
                )
        wandb.log({"evaluation_summary": table})

        # Log overall summary metrics
        all_maes = [
            mets['overall_mae']
            for test_results in evaluation_results.values()
            for mets in test_results.values()
            if not np.isnan(mets['overall_mae'])
        ]
        if all_maes:
            wandb.log({
                "summary/average_mae": np.mean(all_maes),
                "summary/best_mae": np.min(all_maes),
                "summary/worst_mae": np.max(all_maes),
                "summary/num_evaluations": len(all_maes),
            })

    # 7. --- Save final results ---
    class _NumpyEncoder(json.JSONEncoder):
        """Serialise numpy scalars and arrays to native Python types."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results_file = log_dir / "evaluation_summary.json"
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=4, cls=_NumpyEncoder)
    logger.info(f"Full evaluation results saved to {results_file}")

    if use_wandb:
        wandb.finish()
        logger.info("W&B session finished.")

if __name__ == "__main__":
    main() 
    