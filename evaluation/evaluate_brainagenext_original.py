#!/usr/bin/env python
"""
Memory-optimized evaluation script for original BrainAgeNeXt paper weights.

This is a memory-optimized version of evaluate_brainagenext_original.py that:
- Disables dataset caching to reduce memory usage
- Uses lower batch sizes
- Implements memory-efficient data loading
- Reduces multiprocessing to prevent memory leaks
"""

import os, sys, json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

import torchio
import nibabel as nib
from monai.transforms import Compose, ScaleIntensityd, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd, MapTransform
from monai.data import Dataset  # Use regular Dataset instead of CacheDataset

# Fix CUDA multiprocessing issue
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from brain_age_pred.models.create_mednext_encoder_v1 import create_mednext_encoder_v1
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.configs.config import Config


class LoadImageArrayd(MapTransform):
    """Custom MONAI transform to load image arrays (.npy or .nii.gz files)"""
    def __init__(self, keys, ensure_channel_first=True):
        super().__init__(keys)
        self.ensure_channel_first = ensure_channel_first

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], str):
                # Load image array - handle both .npy and .nii.gz formats
                file_path = d[key]
                
                if file_path.endswith(".npy"):
                    try:
                        array = np.load(file_path, allow_pickle=True).astype(np.float32)
                    except (ValueError, OSError):
                        # Fallback: try without allow_pickle for newer files
                        array = np.load(file_path).astype(np.float32)
                elif file_path.endswith(".nii.gz") or file_path.endswith(".nii"):
                    # Load NIfTI file
                    nii_img = nib.load(file_path)
                    array = nii_img.get_fdata().astype(np.float32)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}. Only .npy, .nii, and .nii.gz are supported.")
                
                # Add channel dimension if needed and ensure_channel_first is True
                if self.ensure_channel_first and array.ndim == 3:
                    array = array[np.newaxis, ...]  # Add channel dimension
                d[key] = array
        return d


class MedNeXtEncReg(nn.Module):
    """Original BrainAgeNeXt model architecture using MedNeXt encoder"""
    def __init__(self, *args, **kwargs):
        super(MedNeXtEncReg, self).__init__()
        self.mednextv1 = create_mednext_encoder_v1(
            num_input_channels=1, 
            num_classes=1, 
            model_id='B', 
            kernel_size=3, 
            deep_supervision=True
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regression_fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mednext_out = self.mednextv1(x)
        x = mednext_out
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        age_estimate = self.regression_fc(x)
        return age_estimate.squeeze()


def brain_mask_function(x):
    """Picklable function for brain masking instead of lambda"""
    return x > 0


def prepare_monai_transforms():
    """Prepare MONAI transforms for image arrays - handles both .npy and .nii.gz files"""
    x, y, z = (160, 192, 160)
    p = 1.0
    monai_transforms = [
        LoadImageArrayd(keys=["image"], ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=(p, p, p)),
        CropForegroundd(keys=["image"], allow_smaller=True, source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(x, y, z)),
        CenterSpatialCropd(keys=["image"], roi_size=(x, y, z))
    ]
    val_torchio_transforms = torchio.transforms.Compose(
        [torchio.transforms.ZNormalization(masking_method=brain_mask_function, keys=["image"], include=['image'])]
    )
    return Compose(monai_transforms + [val_torchio_transforms])


def read_csv_with_headmotion_monai(
    csv_path: str,
    data_root: str,
    image_key: str = "image_path",
    age_key: str = "age",
    sex_key: str = "sex",
    modalities_key: str = "modality",
    headmotion_key: str = "headmotion",
):
    """Extended version of read_csv for MONAI dataloader that also extracts headmotion data if available."""
    df = pd.read_csv(csv_path)
    data_dicts = []
    headmotions = []
    sexes = []
    modalities = []
    data_root = Path(data_root)
    
    for _, row in df.iterrows():
        rel_path = row[image_key]
        fpath = data_root / rel_path
        if fpath.exists():
            data_dicts.append({
                'image': str(fpath), 
                'label': float(row[age_key])
            })
            headmotions.append(str(row.get(headmotion_key, 'N/A')))
            sexes.append(str(row.get(sex_key, 'N/A')))
            modalities.append(str(row.get(modalities_key, 'N/A')))
    
    return data_dicts, headmotions, sexes, modalities


def create_monai_dataloader_with_metadata(csv_path, data_dir, batch_size=2, num_workers=2):
    """Create memory-efficient dataloader using MONAI transforms for image arrays (.npy or .nii.gz) with metadata"""
    data_dicts, headmotions, sexes, modalities = read_csv_with_headmotion_monai(csv_path, data_dir)
    
    # Create transforms and dataset
    transforms = prepare_monai_transforms()
    # Use regular Dataset instead of CacheDataset to save memory
    dataset = Dataset(data=data_dicts, transform=transforms)
    
    # Reduced num_workers and disabled pin_memory for memory efficiency
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,  # Reduced from 4 to 2
        shuffle=False, 
        pin_memory=False  # Disabled to save memory
    )
    
    return dataloader, headmotions, sexes, modalities


def load_brainagenext_model(model_path, device):
    """Load BrainAgeNeXt model from checkpoint using original loading approach"""
    print(f"Loading BrainAgeNeXt model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model = MedNeXtEncReg().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"BrainAgeNeXt model loaded successfully on {device}")
    return model


def evaluate_model_brainagenext(model, test_loader, device, headmotions=None, sexes=None, modalities=None):
    """Runs the evaluation loop for BrainAgeNeXt model with comprehensive metadata tracking."""
    model.eval()
    all_preds = []
    all_ages = []
    all_modalities = []
    all_sexes = []
    all_headmotions = []

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            # Clear cache every 10 batches to prevent memory buildup
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
            images = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            
            # Model inference
            pred = model(images)
            
            # Convert to numpy and ensure it's at least 1D for extending the list
            pred_np = pred.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Handle both scalar (0-d) and vector outputs
            if pred_np.ndim == 0:
                all_preds.append(pred_np.item())
            else:
                all_preds.extend(pred_np)
                
            if labels_np.ndim == 0:
                all_ages.append(labels_np.item())
            else:
                all_ages.extend(labels_np)
            
            # Add metadata
            batch_size = len(labels)
            start_idx = i * test_loader.batch_size
            end_idx = start_idx + batch_size
            
            if modalities is not None:
                batch_modalities = modalities[start_idx:end_idx]
                all_modalities.extend(batch_modalities)
            
            if sexes is not None:
                batch_sexes = sexes[start_idx:end_idx]
                all_sexes.extend(batch_sexes)
                
            if headmotions is not None:
                batch_headmotions = headmotions[start_idx:end_idx]
                all_headmotions.extend(batch_headmotions)
            
            # Print progress every 20 batches
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} batches")
            
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
    
    # Overall metrics - convert to native Python types
    mae = float(df['ae'].mean())
    mse = float(df['se'].mean())
    
    # Calculate R² and correlation
    y_true = df['age'].values
    y_pred = df['prediction'].values
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    
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
            mod_mae = float(mod_df['ae'].mean())
            mod_mse = float(mod_df['se'].mean())
            
            mod_y_true = mod_df['age'].values
            mod_y_pred = mod_df['prediction'].values
            
            mod_ss_res = np.sum((mod_y_true - mod_y_pred) ** 2)
            mod_ss_tot = np.sum((mod_y_true - np.mean(mod_y_true)) ** 2)
            mod_r2 = float(1 - (mod_ss_res / mod_ss_tot)) if mod_ss_tot != 0 else 0.0
            
            mod_correlation = float(np.corrcoef(mod_y_true, mod_y_pred)[0, 1]) if len(mod_y_true) > 1 else 0.0
            
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
            sex_mae = float(sex_df['ae'].mean())
            sex_mse = float(sex_df['se'].mean())
            
            sex_y_true = sex_df['age'].values
            sex_y_pred = sex_df['prediction'].values
            
            sex_ss_res = np.sum((sex_y_true - sex_y_pred) ** 2)
            sex_ss_tot = np.sum((sex_y_true - np.mean(sex_y_true)) ** 2)
            sex_r2 = float(1 - (sex_ss_res / sex_ss_tot)) if sex_ss_tot != 0 else 0.0
            
            sex_correlation = float(np.corrcoef(sex_y_true, sex_y_pred)[0, 1]) if len(sex_y_true) > 1 else 0.0
            
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
                hm_mae = float(hm_df['ae'].mean())
                hm_mse = float(hm_df['se'].mean())
                
                hm_y_true = hm_df['age'].values
                hm_y_pred = hm_df['prediction'].values
                
                hm_ss_res = np.sum((hm_y_true - hm_y_pred) ** 2)
                hm_ss_tot = np.sum((hm_y_true - np.mean(hm_y_true)) ** 2)
                hm_r2 = float(1 - (hm_ss_res / hm_ss_tot)) if hm_ss_tot != 0 else 0.0
                
                hm_correlation = float(np.corrcoef(hm_y_true, hm_y_pred)[0, 1]) if len(hm_y_true) > 1 else 0.0
                
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


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="BrainAgeNeXt Original Weights Evaluation (Memory Optimized)")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    # Load configuration
    cfg_file = Path(args.config)
    if not cfg_file.exists():
        print(f"Error: Config file not found at '{cfg_file}'")
        sys.exit(1)
    cfg = Config(cfg_file)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg.get("output_dir", "output/evaluation"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("brainagenext_evaluation", log_file=log_dir / "brainagenext_eval_memopt.log")
    
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")
    
    # Memory optimization settings
    batch_size = max(1, min(cfg.get("batch_size", 2), 2))  # Force max batch size of 2
    num_workers = max(0, min(cfg.get("num_workers", 2), 2))  # Force max num_workers of 2
    logger.info(f"Memory optimized settings: batch_size={batch_size}, num_workers={num_workers}")
    
    # W&B Setup
    use_wandb = cfg.get("wandb.use_wandb", True)
    experiment_name = cfg.get("wandb.experiment_name")
    if not experiment_name:
        experiment_name = f'brainagenext_original_evaluation_memopt_{timestamp}'
    
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
    
    # Evaluation Loop
    evaluation_results = {}
    models_to_eval = cfg.get("models", [])
    test_sets = cfg.get("testing", [])
    global_data_dir = Path(cfg.get("data_dir"))
    
    if not models_to_eval or not test_sets:
        logger.error("Config file must contain 'models' and 'testing' sections.")
        return
    
    for model_info in models_to_eval:
        model_name = model_info["name"]
        logger.info(f"--- Evaluating BrainAgeNeXt model: {model_name} ---")
        evaluation_results[model_name] = {}
        
        # Load BrainAgeNeXt model
        checkpoint_path = model_info["checkpoint"]
        try:
            model = load_brainagenext_model(checkpoint_path, device)
        except Exception as e:
            logger.error(f"Could not load checkpoint for model '{model_name}': {e}. Skipping.")
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
            
            # Clear GPU cache before creating dataloader
            torch.cuda.empty_cache()
            
            # Create MONAI dataloader with metadata (memory optimized)
            test_loader, headmotions, sexes, modalities = create_monai_dataloader_with_metadata(
                test_csv, data_dir, 
                batch_size=batch_size,
                num_workers=num_workers
            )
            
            # Evaluate and calculate metrics
            preds, true_ages, mods, sxs, hmotions = evaluate_model_brainagenext(
                model, test_loader, device, headmotions, sexes, modalities
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
                
                # Log headmotion-specific metrics if available
                if metrics.get('headmotion_metrics'):
                    for headmotion, hm_metrics in metrics['headmotion_metrics'].items():
                        wandb.log({
                            f"{log_prefix}/{headmotion}_mae": hm_metrics['mae'],
                            f"{log_prefix}/{headmotion}_mse": hm_metrics['mse'],
                            f"{log_prefix}/{headmotion}_r2": hm_metrics['r2'],
                            f"{log_prefix}/{headmotion}_correlation": hm_metrics['correlation'],
                            f"{log_prefix}/{headmotion}_count": hm_metrics['count']
                        })
            
            # Print summary
            print_summary_table(model_name, test_set_name, metrics)
            
            # Clear cache after each test set
            torch.cuda.empty_cache()
    
    # Final Summary and W&B Table
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
    
    # Save final results
    results_file = log_dir / "brainagenext_evaluation_summary_memopt.json"
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    logger.info(f"Full evaluation results saved to {results_file}")
    
    if use_wandb:
        wandb.finish()
        logger.info("W&B session finished")


if __name__ == "__main__":
    main()