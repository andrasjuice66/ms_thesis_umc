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

import pandas as pd
import numpy as np
import torch
import wandb
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
    
    metrics = {
        'overall_mae': df['ae'].mean()
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

def print_summary_table(model_name, test_set_name, metrics):
    """Prints a formatted summary table of the evaluation metrics."""
    print("\n" + "="*80)
    print(f"Evaluation Summary: Model '{model_name}' on Test Set '{test_set_name}'")
    print("-"*80)
    
    print(f"  Overall MAE: {metrics['overall_mae']:.4f}")
    
    print("\n  MAE by Modality:")
    if metrics['modality_mae']:
        for modality, mae in metrics['modality_mae'].items():
            print(f"    - {modality}: {mae:.4f}")
    else:
        print("    No modality data available.")
        
    print("\n  MAE by Sex:")
    if metrics['sex_mae']:
        for sex, mae in metrics['sex_mae'].items():
            print(f"    - {sex}: {mae:.4f}")
    else:
        print("    No sex data available.")
    
    # Add headmotion results if available
    if 'headmotion_mae' in metrics:
        print("\n  MAE by Head Motion Type:")
        for headmotion, mae in metrics['headmotion_mae'].items():
            print(f"    - {headmotion}: {mae:.4f}")
        
    print("="*80 + "\n")

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

            # Load data with headmotion support
            paths, ages, _, sexes, modalities, headmotions = read_csv_with_headmotion(
                test_csv,
                data_dir,
            )

            # Check if this model requires segmentation input
            model_params = model_info["params"]
            in_channels = model_params.get("in_channels", 1)
            should_normalize = model_params.get("normalize", True)

            if in_channels > 1:
                # Model expects multi-channel segmentation input (one-hot encoded)
                transform = SegmentationTestGenerator(n_classes=in_channels)
                logger.info(f"Using segmentation transforms with {in_channels} classes.")
                # Disable BADataset's built-in normalization/clamping — the transform handles it
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

            # Evaluate and calculate metrics
            preds, true_ages, mods, sxs, hmotions = evaluate_model(
                model, test_loader, device, model_info["params"]["type"], headmotions
            )
            metrics = calculate_metrics(preds, true_ages, mods, sxs, hmotions)
            
            evaluation_results[model_name][test_set_name] = metrics
            
            # Print summary
            print_summary_table(model_name, test_set_name, metrics)

            # Log to W&B
            if use_wandb:
                log_prefix = f"{model_name}/{test_set_name}"
                wandb_log = {
                    f"{log_prefix}/overall_mae": metrics['overall_mae'],
                }
                # Log per-modality MAE
                for mod, mae in metrics.get('modality_mae', {}).items():
                    wandb_log[f"{log_prefix}/modality/{mod}_mae"] = mae
                # Log per-sex MAE
                for sex, mae in metrics.get('sex_mae', {}).items():
                    wandb_log[f"{log_prefix}/sex/{sex}_mae"] = mae
                # Log per-headmotion MAE
                for hm, mae in metrics.get('headmotion_mae', {}).items():
                    wandb_log[f"{log_prefix}/headmotion/{hm}_mae"] = mae
                wandb.log(wandb_log)

    # 5. --- W&B Summary Table ---
    if use_wandb:
        # Build a summary table with all results
        columns = ["Model", "Test Set", "Overall MAE", "Modality MAEs", "Sex MAEs"]
        table = wandb.Table(columns=columns)
        for m_name, test_results in evaluation_results.items():
            for ts_name, mets in test_results.items():
                mod_str = ", ".join(f"{k}: {v:.4f}" for k, v in mets.get('modality_mae', {}).items())
                sex_str = ", ".join(f"{k}: {v:.4f}" for k, v in mets.get('sex_mae', {}).items())
                table.add_data(m_name, ts_name, f"{mets['overall_mae']:.4f}", mod_str, sex_str)
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

    # 6. --- Save final results ---
    results_file = log_dir / "evaluation_summary.json"
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    logger.info(f"Full evaluation results saved to {results_file}")

    if use_wandb:
        wandb.finish()
        logger.info("W&B session finished.")

if __name__ == "__main__":
    main() 
    