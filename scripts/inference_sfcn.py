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

# -------- Set multiprocessing start method BEFORE other imports --------
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

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

from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer
from brain_age_pred.training.metrics import calculate_metrics
from brain_age_pred.utils.utils import read_csv
from brain_age_pred.configs.config import Config

# Import utility functions from SFCN model


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
MODEL_PATH = os.path.join(MODEL_DIR, 'sfcn_original_ckp.p')

TEST_CSV    = '/home/ajoos/brain_age_pred/data/labels/test_balanced.csv'
DATA_ROOT   = '/scratch-shared/ajoos/'

# Config file path
CONFIG_PATH = str('/home/ajoos/brain_age_pred/configs/evaluate/sfcn_original.yaml')

OUT_DIR     = Path('.')
OUT_DIR.mkdir(exist_ok=True)

# --- ❸ runtime -------------------------------------------------------------
BATCH_SIZE  = 8
NUM_WORKERS = 4
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WANDB_ENABLED = True          # switch off if you do not want Weights&Biases
WANDB_API = '2abdb867a9244072f2237704a3cacc77fa548dd8'

# --- ❹ evaluation settings -------------------------------------------------
N_FOLDS = 10  # for domain randomization evaluations
SIGMA = 1     # for soft label conversion

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


# ======================= UTILITY FUNCTIONS ===============================

def predict_age_from_probabilities(log_probs, bin_centres):
    """
    Convert log probabilities to predicted age using weighted sum.
    
    Args:
        log_probs: Log probabilities from model output [batch_size, n_bins] or [n_bins]
        bin_centres: Age bin centers [n_bins]
    
    Returns:
        Predicted ages [batch_size] or scalar
    """
    probs = torch.exp(log_probs)  # Convert log probs to probs
    
    # Handle both batched and single sample cases
    if len(probs.shape) == 1:
        # Single sample case: [n_bins]
        predicted_ages = torch.sum(probs * bin_centres)
    else:
        # Batch case: [batch_size, n_bins]
        predicted_ages = torch.sum(probs * bin_centres, dim=1)
    
    return predicted_ages

def calculate_mae_by_modality(predictions, targets, modalities, dataset_name=""):
    """Calculate MAE overall and per modality."""
    overall_mae = torch.mean(torch.abs(predictions - targets)).item()
    
    results = {
        f'{dataset_name}_mae_overall': overall_mae,
        f'{dataset_name}_n_samples': len(predictions)
    }
    
    # Calculate per modality if modalities are available
    if modalities is not None:
        unique_modalities = set(modalities)
        for modality in unique_modalities:
            mask = [mod.upper() == modality.upper() for mod in modalities]
            if any(mask):
                mod_predictions = predictions[mask]
                mod_targets = targets[mask]
                mod_mae = torch.mean(torch.abs(mod_predictions - mod_targets)).item()
                results[f'{dataset_name}_mae_{modality.lower()}'] = mod_mae
                results[f'{dataset_name}_n_{modality.lower()}'] = len(mod_predictions)
    
    return results


def load_model(model_path, device):
    """Load SFCN model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    # Initialize model
    model = SFCN(output_dim=N_BINS, dropout=True)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load with proper device mapping
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def load_test_data(csv_path, data_root):
    """Load test dataset from CSV and filter by age range."""
    print(f"Loading test data from {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} samples in original test set")
    
    # Extract required columns
    required_cols = ['image_path', 'age']
    optional_cols = ['modality', 'sex']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    # Filter by age range (42 to 82, inclusive)
    print(f"Filtering samples to age range: {BIN_RANGE[0]} to {BIN_RANGE[1]} years")
    age_mask = (df['age'] >= BIN_RANGE[0]) & (df['age'] <= BIN_RANGE[1])
    df_filtered = df[age_mask].copy()
    
    print(f"After age filtering: {len(df_filtered)} samples remaining")
    print(f"Filtered out: {len(df) - len(df_filtered)} samples outside age range")
    
    if len(df_filtered) == 0:
        raise ValueError(f"No samples found within age range {BIN_RANGE[0]}-{BIN_RANGE[1]}")
    
    # Print age distribution
    print(f"Age range in filtered data: {df_filtered['age'].min():.1f} to {df_filtered['age'].max():.1f} years")
    print(f"Mean age: {df_filtered['age'].mean():.1f} ± {df_filtered['age'].std():.1f} years")
    
    # Construct full paths
    file_paths = [os.path.join(data_root, path) for path in df_filtered['image_path']]
    ages = df_filtered['age'].tolist()
    
    # Optional columns
    modalities = df_filtered['modality'].tolist() if 'modality' in df_filtered.columns else None
    sexes = df_filtered['sex'].tolist() if 'sex' in df_filtered.columns else None
    
    # Print modality distribution if available
    if modalities:
        modality_counts = df_filtered['modality'].value_counts()
        print("Modality distribution:")
        for mod, count in modality_counts.items():
            print(f"  {mod}: {count} samples")
    
    # Verify files exist
    missing_files = [path for path in file_paths if not os.path.exists(path)]
    if missing_files:
        print(f"Warning: {len(missing_files)} files not found")
        for f in missing_files[:5]:  # Show first 5
            print(f"  Missing: {f}")
    
    return file_paths, ages, modalities, sexes

def create_domain_randomizer(config, use_tumor=False):
    """Create domain randomizer with parameters from config file."""
    
    # Get domain randomization config from file
    dr_config_from_file = config.get('domain_randomization', {})
    
    # Base domain randomization config - make a COPY of transform_probs to avoid reference issues
    transform_probs = dr_config_from_file.get('transform_probs', {}).copy()
    
    dr_config = {
        'device': DEVICE,
        'use_domain_randomization': True,
        'transform_probs': transform_probs,
        'output_shape': dr_config_from_file.get('output_shape', [160, 192, 160]),
        'use_tumor_simulation': use_tumor,
    }
    
    # Override tumor probability based on use_tumor parameter
    if 'tumor' in transform_probs:
        original_tumor_prob = dr_config_from_file['transform_probs']['tumor']
        dr_config['transform_probs']['tumor'] = original_tumor_prob if use_tumor else 0.0
    
    # Add all other domain randomization parameters from config
    for key, value in dr_config_from_file.items():
        if key not in ['transform_probs', 'output_shape']:  # Don't override already set keys
            dr_config[key] = value
    
    # Add tumor config if needed
    if use_tumor and 'tumor_config' in dr_config_from_file:
        dr_config['tumor_config'] = dr_config_from_file['tumor_config']
    
    return DomainRandomizer(**dr_config)


def evaluate_regime(model, dataset, dataloader, regime_name, fold_idx=None):
    """Evaluate model on a dataset regime."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_modalities = []
    
    print(f"Evaluating {regime_name}" + (f" (fold {fold_idx+1})" if fold_idx is not None else ""))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx+1}/{len(dataloader)}")
            
            # Get image and age
            image = batch['image'].to(DEVICE)
            age = batch['age'].to(DEVICE)
            
            # Get modality if available
            modality = batch.get('modality', [None] * len(age))
            
            # Model inference
            outputs = model(image)
            log_probs = outputs[0]  # Don't squeeze yet: [batch_size, n_bins, 1, 1, 1]
            
            # Remove spatial dimensions but keep batch dimension
            log_probs = log_probs.squeeze(-1).squeeze(-1).squeeze(-1)  # [batch_size, n_bins]
            
            # Debug print to check dimensions
            #print(f"  log_probs shape: {log_probs.shape}, bin_centres shape: {torch.tensor(bin_centres).shape}")
            
            # Convert to predicted ages
            pred_ages = predict_age_from_probabilities(log_probs, torch.tensor(bin_centres).to(DEVICE))
            
            # Ensure pred_ages is always a tensor with batch dimension
            if len(pred_ages.shape) == 0:  # scalar
                pred_ages = pred_ages.unsqueeze(0)
            
            # Store results
            all_predictions.append(pred_ages.cpu())
            all_targets.append(age.cpu())
            all_modalities.extend(modality)
    
    # Concatenate all results
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    # Calculate metrics
    metrics = calculate_mae_by_modality(
        predictions, targets, all_modalities, 
        dataset_name=regime_name + (f"_fold{fold_idx+1}" if fold_idx is not None else "")
    )
    
    return metrics, predictions.numpy(), targets.numpy()


def create_metrics_table(metrics, table_name, evaluation_type):
    """Create a W&B table from metrics dictionary - same as BrainAgeNeXt"""
    # Extract overall metrics
    overall_metrics = {
        "Metric": ["MAE"],
        "Value": [metrics.get("mae_overall", metrics.get("mae", 0))]
    }
    
    # Add standard deviation if available (for multi-fold evaluations)
    if "mae_std" in metrics:
        overall_metrics["Std"] = [metrics.get("mae_std", 0)]
    
    # Create overall table
    overall_table = wandb.Table(columns=list(overall_metrics.keys()))
    for i in range(len(overall_metrics["Metric"])):
        row = [overall_metrics[key][i] for key in overall_metrics.keys()]
        overall_table.add_data(*row)
    
    # Create modality-specific table if modality metrics exist
    modality_data = []
    modality_keys = [k for k in metrics.keys() if "_mae_" in k and not k.endswith("_std")]
    modalities_found = set()
    
    for key in modality_keys:
        if "_mae_" in key:
            # Extract modality name (e.g., from "normal_test_mae_t1" get "t1")
            parts = key.split("_mae_")
            if len(parts) == 2:
                modality = parts[1]
                modalities_found.add(modality)
    
    for modality in modalities_found:
        mae_key = f"{table_name.replace('_test', '')}_mae_{modality}"
        mae_val = metrics.get(mae_key, 0)
        
        # Add standard deviation if available
        if f"{mae_key}_std" in metrics:
            mae_std = metrics.get(f"{mae_key}_std", 0)
            modality_data.append([modality.upper(), f"{mae_val:.4f} ± {mae_std:.4f}"])
        else:
            modality_data.append([modality.upper(), f"{mae_val:.4f}"])
    
    modality_table = None
    if modality_data:
        columns = ["Modality", "MAE"]
        modality_table = wandb.Table(columns=columns)
        for row in modality_data:
            modality_table.add_data(*row)
    
    # Log tables to W&B
    wandb.log({f"{table_name}_overall_metrics": overall_table})
    if modality_table:
        wandb.log({f"{table_name}_modality_metrics": modality_table})
    
    return overall_table, modality_table

def convert_sfcn_metrics_to_standard_format(sfcn_metrics, regime_name):
    """Convert SFCN metric format to standard format matching BrainAgeNeXt"""
    standard_metrics = {}
    
    # Extract overall MAE
    mae_overall_key = f"{regime_name}_mae_overall"
    if mae_overall_key in sfcn_metrics:
        standard_metrics["mae"] = sfcn_metrics[mae_overall_key]
    
    # Extract modality-specific MAEs
    for key, value in sfcn_metrics.items():
        if f"{regime_name}_mae_" in key and not key.endswith("_overall"):
            # Extract modality name
            modality = key.replace(f"{regime_name}_mae_", "")
            standard_metrics[f"{modality}_mae"] = value
            
    # Extract sample counts
    for key, value in sfcn_metrics.items():
        if f"{regime_name}_n_" in key:
            modality = key.replace(f"{regime_name}_n_", "")
            standard_metrics[f"{modality}_n"] = value
    
    return standard_metrics

def main():
    """Main inference function."""
    print("Starting SFCN Inference Script")
    print(f"Device: {DEVICE}")
    print(f"Bin range: {BIN_RANGE}, step: {BIN_STEP}, bins: {N_BINS}")
    print("-" * 60)
    
    # Load configuration
    print(f"Loading configuration from: {CONFIG_PATH}")
    config = Config(CONFIG_PATH)
    
    # Initialize wandb with better structure like BrainAgeNeXt
    if WANDB_ENABLED:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"sfcn_3regime_eval_{timestamp}"
        
        wandb.login(key=WANDB_API)
        wandb.init(
            project="brainage-inference",
            name=experiment_name,
            config={
                'model': 'SFCN',
                'model_path': MODEL_PATH,
                'test_csv': TEST_CSV,
                'config_path': CONFIG_PATH,
                'bin_range': BIN_RANGE,
                'bin_step': BIN_STEP,
                'n_bins': N_BINS,
                'n_folds': N_FOLDS,
                'device': str(DEVICE),
                'batch_size': BATCH_SIZE,
                'evaluation_type': '3regime_inference',
                'domain_randomization_config': config.get('domain_randomization', {}),
            },
            reinit=True,
        )
    
    # Load model
    model = load_model(MODEL_PATH, DEVICE)
    
    # Load test data
    file_paths, ages, modalities, sexes = load_test_data(TEST_CSV, DATA_ROOT)
    
    # Log dataset info to W&B like BrainAgeNeXt
    if WANDB_ENABLED:
        wandb.log({
            "dataset/num_samples": len(ages),
            "dataset/age_min": min(ages),
            "dataset/age_max": max(ages),
            "dataset/age_mean": np.mean(ages),
            "dataset/age_std": np.std(ages),
        })
    
    all_results = {}
    
    # ======================== REGIME 1: Normal Test ========================
    print("\n" + "="*60)
    print("REGIME 1: Normal Test (No Augmentation)")
    print("="*60)
    
    # Create normal dataset (no transforms)
    normal_dataset = BADataset(
        file_paths=file_paths,
        age_labels=ages,
        modalities=modalities,
        sexes=sexes,
        transform=None,  # No augmentation
        mode='test'
    )
    
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Evaluate normal test
    metrics, predictions, targets = evaluate_regime(model, normal_dataset, normal_loader, "normal_test")
    all_results.update(metrics)
    
    # Convert to standard format and log to W&B
    normal_standard_metrics = convert_sfcn_metrics_to_standard_format(metrics, "normal_test")
    if WANDB_ENABLED:
        wandb.log({f"test/{k}": v for k, v in normal_standard_metrics.items()})
        create_metrics_table(normal_standard_metrics, "normal_test", "Normal Test")
    
    print(f"Normal Test Results:")
    print(f"  Overall MAE: {metrics['normal_test_mae_overall']:.3f}")
    if modalities:
        for mod in ['t1', 't2', 'flair']:
            if f'normal_test_mae_{mod}' in metrics:
                print(f"  {mod.upper()} MAE: {metrics[f'normal_test_mae_{mod}']:.3f} (n={metrics[f'normal_test_n_{mod}']})")
    
    # ================= REGIME 2: Domain Randomization ==================
    print("\n" + "="*60)
    print("REGIME 2: Domain Randomization (10 folds)")
    print("="*60)
    
    dr_results = []
    for fold in range(N_FOLDS):
        print(f"\nFold {fold+1}/{N_FOLDS}")
        
        # Create domain randomizer (no tumor) with config parameters
        domain_randomizer = create_domain_randomizer(config, use_tumor=False)
        
        # Create dataset with domain randomization
        dr_dataset = BADataset(
            file_paths=file_paths,
            age_labels=ages,
            modalities=modalities,
            sexes=sexes,
            transform=domain_randomizer,
            mode='train'  # Apply transforms
        )
        
        dr_loader = DataLoader(
            dr_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        # Evaluate domain randomization
        metrics, predictions, targets = evaluate_regime(
            model, dr_dataset, dr_loader, "domain_rand", fold_idx=fold
        )
        dr_results.append(metrics)
        all_results.update(metrics)
    
    # Aggregate domain randomization results
    dr_maes = [result[f'domain_rand_fold{i+1}_mae_overall'] for i, result in enumerate(dr_results)]
    dr_mean_mae = np.mean(dr_maes)
    dr_std_mae = np.std(dr_maes)
    
    # Create aggregated metrics in standard format
    dr_standard_metrics = {"mae": dr_mean_mae, "mae_std": dr_std_mae}
    
    # Per-modality aggregation
    if modalities:
        for mod in ['t1', 't2', 'flair']:
            mod_maes = []
            for i, result in enumerate(dr_results):
                key = f'domain_rand_fold{i+1}_mae_{mod}'
                if key in result:
                    mod_maes.append(result[key])
            if mod_maes:
                mod_mean = np.mean(mod_maes)
                mod_std = np.std(mod_maes)
                dr_standard_metrics[f'{mod}_mae'] = mod_mean
                dr_standard_metrics[f'{mod}_mae_std'] = mod_std
    
    # Log to W&B
    if WANDB_ENABLED:
        wandb.log({f"test_dom_rand/{k}": v for k, v in dr_standard_metrics.items()})
        create_metrics_table(dr_standard_metrics, "domain_randomized_test", "Domain Randomized Test")
    
    all_results['domain_rand_mae_mean'] = dr_mean_mae
    all_results['domain_rand_mae_std'] = dr_std_mae
    
    print(f"\nDomain Randomization Results (10 folds):")
    print(f"  Mean MAE: {dr_mean_mae:.3f} ± {dr_std_mae:.3f}")
    
    # ========== REGIME 3: Domain Randomization + Tumor Simulation ==========
    print("\n" + "="*60)
    print("REGIME 3: Domain Randomization + Tumor Simulation (10 folds)")
    print("="*60)
    
    tumor_results = []
    for fold in range(N_FOLDS):
        print(f"\nFold {fold+1}/{N_FOLDS}")
        
        # Create domain randomizer with tumor simulation using config parameters
        domain_randomizer_tumor = create_domain_randomizer(config, use_tumor=True)
        
        # Create dataset with domain randomization + tumor
        tumor_dataset = BADataset(
            file_paths=file_paths,
            age_labels=ages,
            modalities=modalities,
            sexes=sexes,
            transform=domain_randomizer_tumor,
            mode='train'  # Apply transforms
        )
        
        tumor_loader = DataLoader(
            tumor_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        # Evaluate domain randomization + tumor
        metrics, predictions, targets = evaluate_regime(
            model, tumor_dataset, tumor_loader, "domain_rand_tumor", fold_idx=fold
        )
        tumor_results.append(metrics)
        all_results.update(metrics)
    
    # Aggregate tumor simulation results
    tumor_maes = [result[f'domain_rand_tumor_fold{i+1}_mae_overall'] for i, result in enumerate(tumor_results)]
    tumor_mean_mae = np.mean(tumor_maes)
    tumor_std_mae = np.std(tumor_maes)
    
    # Create aggregated metrics in standard format
    tumor_standard_metrics = {"mae": tumor_mean_mae, "mae_std": tumor_std_mae}
    
    # Per-modality aggregation
    if modalities:
        for mod in ['t1', 't2', 'flair']:
            mod_maes = []
            for i, result in enumerate(tumor_results):
                key = f'domain_rand_tumor_fold{i+1}_mae_{mod}'
                if key in result:
                    mod_maes.append(result[key])
            if mod_maes:
                mod_mean = np.mean(mod_maes)
                mod_std = np.std(mod_maes)
                tumor_standard_metrics[f'{mod}_mae'] = mod_mean
                tumor_standard_metrics[f'{mod}_mae_std'] = mod_std
    
    # Log to W&B
    if WANDB_ENABLED:
        wandb.log({f"test_dom_rand_tumor/{k}": v for k, v in tumor_standard_metrics.items()})
        create_metrics_table(tumor_standard_metrics, "domain_rand_tumor_test", "Domain Randomized + Tumor Test")
    
    all_results['domain_rand_tumor_mae_mean'] = tumor_mean_mae
    all_results['domain_rand_tumor_mae_std'] = tumor_std_mae
    
    print(f"\nDomain Randomization + Tumor Results (10 folds):")
    print(f"  Mean MAE: {tumor_mean_mae:.3f} ± {tumor_std_mae:.3f}")
    
    # ======================== SUMMARY ========================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"1. Normal Test MAE:           {all_results['normal_test_mae_overall']:.3f}")
    print(f"2. Domain Randomization MAE:  {dr_mean_mae:.3f} ± {dr_std_mae:.3f}")
    print(f"3. Domain Rand + Tumor MAE:   {tumor_mean_mae:.3f} ± {tumor_std_mae:.3f}")
    
    # Log summary comparison to W&B like BrainAgeNeXt
    if WANDB_ENABLED:
        wandb.log({
            "evaluation_summary/normal_mae": all_results['normal_test_mae_overall'],
            "evaluation_summary/dom_rand_mae": dr_mean_mae,
            "evaluation_summary/dom_rand_tumor_mae": tumor_mean_mae,
            "evaluation_summary/dom_rand_mae_std": dr_std_mae,
            "evaluation_summary/dom_rand_tumor_mae_std": tumor_std_mae,
        })
        
        # Create visualization like BrainAgeNeXt
        plt.figure(figsize=(12, 4))
        
        # Plot 1: MAE comparison
        plt.subplot(1, 2, 1)
        mae_values = [all_results['normal_test_mae_overall'], dr_mean_mae, tumor_mean_mae]
        mae_stds = [0, dr_std_mae, tumor_std_mae]
        labels = ['Normal', 'Domain Rand', 'Dom Rand + Tumor']
        plt.bar(labels, mae_values, yerr=mae_stds, capsize=5)
        plt.ylabel('MAE')
        plt.title('MAE Comparison (SFCN)')
        plt.grid(axis='y')
        
        # Plot 2: Summary table visualization
        plt.subplot(1, 2, 2)
        plt.axis('tight')
        plt.axis('off')
        table_data = [
            ['Normal Test', f"{all_results['normal_test_mae_overall']:.3f}", ''],
            ['Domain Randomization', f"{dr_mean_mae:.3f}", f"± {dr_std_mae:.3f}"],
            ['Dom Rand + Tumor', f"{tumor_mean_mae:.3f}", f"± {tumor_std_mae:.3f}"]
        ]
        table = plt.table(cellText=table_data, colLabels=['Regime', 'MAE', 'Std'], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        plt.title('SFCN Evaluation Summary')
        
        plt.tight_layout()
        plt.savefig('sfcn_3regime_evaluation_comparison.png', dpi=300, bbox_inches='tight')
        
        # Log the plot to W&B
        wandb.log({"evaluation_plots": wandb.Image('sfcn_3regime_evaluation_comparison.png')})
        plt.close()
        
        wandb.finish()
    
    # Save results to file
    results_file = OUT_DIR / f'sfcn_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()