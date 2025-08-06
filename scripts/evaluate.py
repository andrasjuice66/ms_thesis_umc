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
from brain_age_pred.brain_gen.labels import GENERATION_LABELS

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
            n_classes=GENERATION_LABELS,
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

def calculate_metrics(predictions, true_ages, modalities, sexes, headmotions=None):
    """Calculates MAE, MSE, R², and correlation overall, per modality, per sex, and per headmotion type."""
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
    
    # Overall metrics
    mae = df['ae'].mean()
    mse = df['se'].mean()
    
    # Calculate R² and correlation
    y_true = df['age'].values
    y_pred = df['prediction'].values
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    
    metrics = {
        'overall_mae': mae,
        'overall_mse': mse,
        'overall_r2': r2,
        'overall_correlation': correlation,
        'count': len(df)
    }
    
    # MAE per modality
    modality_metrics = {}
    for modality in df['modality'].unique():
        mod_df = df[df['modality'] == modality]
        if len(mod_df) > 0:
            mod_mae = mod_df['ae'].mean()
            mod_mse = mod_df['se'].mean()
            
            mod_y_true = mod_df['age'].values
            mod_y_pred = mod_df['prediction'].values
            
            mod_ss_res = np.sum((mod_y_true - mod_y_pred) ** 2)
            mod_ss_tot = np.sum((mod_y_true - np.mean(mod_y_true)) ** 2)
            mod_r2 = 1 - (mod_ss_res / mod_ss_tot) if mod_ss_tot != 0 else 0
            
            mod_correlation = np.corrcoef(mod_y_true, mod_y_pred)[0, 1] if len(mod_y_true) > 1 else 0
            
            modality_metrics[modality] = {
                'mae': mod_mae,
                'mse': mod_mse,
                'r2': mod_r2,
                'correlation': mod_correlation,
                'count': len(mod_df)
            }
    
    metrics['modality_metrics'] = modality_metrics
    
    # MAE per sex
    sex_metrics = {}
    for sex in df['sex'].unique():
        sex_df = df[df['sex'] == sex]
        if len(sex_df) > 0:
            sex_mae = sex_df['ae'].mean()
            sex_mse = sex_df['se'].mean()
            
            sex_y_true = sex_df['age'].values
            sex_y_pred = sex_df['prediction'].values
            
            sex_ss_res = np.sum((sex_y_true - sex_y_pred) ** 2)
            sex_ss_tot = np.sum((sex_y_true - np.mean(sex_y_true)) ** 2)
            sex_r2 = 1 - (sex_ss_res / sex_ss_tot) if sex_ss_tot != 0 else 0
            
            sex_correlation = np.corrcoef(sex_y_true, sex_y_pred)[0, 1] if len(sex_y_true) > 1 else 0
            
            sex_metrics[sex] = {
                'mae': sex_mae,
                'mse': sex_mse,
                'r2': sex_r2,
                'correlation': sex_correlation,
                'count': len(sex_df)
            }
    
    metrics['sex_metrics'] = sex_metrics
    
    # MAE per headmotion type (if available)
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
                hm_mae = hm_df['ae'].mean()
                hm_mse = hm_df['se'].mean()
                
                hm_y_true = hm_df['age'].values
                hm_y_pred = hm_df['prediction'].values
                
                hm_ss_res = np.sum((hm_y_true - hm_y_pred) ** 2)
                hm_ss_tot = np.sum((hm_y_true - np.mean(hm_y_true)) ** 2)
                hm_r2 = 1 - (hm_ss_res / hm_ss_tot) if hm_ss_tot != 0 else 0
                
                hm_correlation = np.corrcoef(hm_y_true, hm_y_pred)[0, 1] if len(hm_y_true) > 1 else 0
                
                readable_name = headmotion_mapping.get(str(headmotion), str(headmotion))
                headmotion_metrics[readable_name] = {
                    'mae': hm_mae,
                    'mse': hm_mse,
                    'r2': hm_r2,
                    'correlation': hm_correlation,
                    'count': len(hm_df)
                }
        
        metrics['headmotion_metrics'] = headmotion_metrics
    
    return metrics

def print_summary_table(model_name, test_set_name, metrics):
    """Prints a formatted summary table of the evaluation metrics."""
    print("\n" + "="*80)
    print(f"Evaluation Summary: Model '{model_name}' on Test Set '{test_set_name}'")
    print("-"*80)
    
    print(f"  Overall MAE: {metrics['overall_mae']:.4f}")
    print(f"  Overall MSE: {metrics['overall_mse']:.4f}")
    print(f"  Overall R²:  {metrics['overall_r2']:.4f}")
    print(f"  Overall Correlation: {metrics['overall_correlation']:.4f}")
    print(f"  Sample Count: {metrics['count']}")
    
    print("\n  Metrics by Modality:")
    if metrics.get('modality_metrics'):
        for modality, mod_metrics in metrics['modality_metrics'].items():
            print(f"    - {modality} (n={mod_metrics['count']}): MAE={mod_metrics['mae']:.4f}, "
                  f"MSE={mod_metrics['mse']:.4f}, R²={mod_metrics['r2']:.4f}, "
                  f"Corr={mod_metrics['correlation']:.4f}")
    else:
        print("    No modality data available.")
        
    print("\n  Metrics by Sex:")
    if metrics.get('sex_metrics'):
        for sex, sex_metrics_data in metrics['sex_metrics'].items():
            print(f"    - {sex} (n={sex_metrics_data['count']}): MAE={sex_metrics_data['mae']:.4f}, "
                  f"MSE={sex_metrics_data['mse']:.4f}, R²={sex_metrics_data['r2']:.4f}, "
                  f"Corr={sex_metrics_data['correlation']:.4f}")
    else:
        print("    No sex data available.")
    
    # Add headmotion results if available
    if 'headmotion_metrics' in metrics:
        print("\n  Metrics by Head Motion Type:")
        for headmotion, hm_metrics in metrics['headmotion_metrics'].items():
            print(f"    - {headmotion} (n={hm_metrics['count']}): MAE={hm_metrics['mae']:.4f}, "
                  f"MSE={hm_metrics['mse']:.4f}, R²={hm_metrics['r2']:.4f}, "
                  f"Corr={hm_metrics['correlation']:.4f}")
        
    print("="*80 + "\n")

def create_wandb_summary_table(evaluation_results):
    """Create a comprehensive wandb table showing all evaluation results."""
    table_data = []
    
    for model_name, test_results in evaluation_results.items():
        for test_set_name, metrics in test_results.items():
            # Overall metrics row
            table_data.append([
                model_name,
                test_set_name,
                "Overall",
                "N/A",
                metrics['count'],
                f"{metrics['overall_mae']:.4f}",
                f"{metrics['overall_mse']:.4f}",
                f"{metrics['overall_r2']:.4f}",
                f"{metrics['overall_correlation']:.4f}"
            ])
            
            # Modality-specific rows
            if metrics.get('modality_metrics'):
                for modality, mod_metrics in metrics['modality_metrics'].items():
                    table_data.append([
                        model_name,
                        test_set_name,
                        "Modality",
                        modality,
                        mod_metrics['count'],
                        f"{mod_metrics['mae']:.4f}",
                        f"{mod_metrics['mse']:.4f}",
                        f"{mod_metrics['r2']:.4f}",
                        f"{mod_metrics['correlation']:.4f}"
                    ])
            
            # Sex-specific rows
            if metrics.get('sex_metrics'):
                for sex, sex_metrics_data in metrics['sex_metrics'].items():
                    table_data.append([
                        model_name,
                        test_set_name,
                        "Sex",
                        sex,
                        sex_metrics_data['count'],
                        f"{sex_metrics_data['mae']:.4f}",
                        f"{sex_metrics_data['mse']:.4f}",
                        f"{sex_metrics_data['r2']:.4f}",
                        f"{sex_metrics_data['correlation']:.4f}"
                    ])
            
            # Headmotion-specific rows
            if metrics.get('headmotion_metrics'):
                for headmotion, hm_metrics in metrics['headmotion_metrics'].items():
                    table_data.append([
                        model_name,
                        test_set_name,
                        "HeadMotion",
                        headmotion,
                        hm_metrics['count'],
                        f"{hm_metrics['mae']:.4f}",
                        f"{hm_metrics['mse']:.4f}",
                        f"{hm_metrics['r2']:.4f}",
                        f"{hm_metrics['correlation']:.4f}"
                    ])
    
    # Create wandb table
    table = wandb.Table(
        columns=[
            "Model", "Test Set", "Category", "Subcategory", "Count",
            "MAE", "MSE", "R²", "Correlation"
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
                'Count': metrics['count']
            })
    
    # Print as formatted table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(f"\n{'Model':<20} {'Test Set':<20} {'MAE':<8} {'MSE':<8} {'R²':<8} {'Corr':<8} {'Count':<8}")
        print("-" * 100)
        for _, row in df.iterrows():
            print(f"{row['Model']:<20} {row['Test Set']:<20} {row['MAE']:<8.4f} {row['MSE']:<8.4f} "
                  f"{row['R²']:<8.4f} {row['Correlation']:<8.4f} {row['Count']:<8}")
        
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

            test_ds = BADataset(
                file_paths=paths,
                age_labels=ages,
                sexes=sexes,
                modalities=modalities,
                mode="test",
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
            
            # Log to W&B
            if use_wandb:
                log_prefix = f"{model_name}_{test_set_name}"
                wandb.log({
                    f"{log_prefix}/overall_mae": metrics['overall_mae'],
                    f"{log_prefix}/overall_mse": metrics['overall_mse'], 
                    f"{log_prefix}/overall_r2": metrics['overall_r2'],
                    f"{log_prefix}/overall_correlation": metrics['overall_correlation'],
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
                            f"{log_prefix}/{sex}_count": sex_metrics_data['count']
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
    