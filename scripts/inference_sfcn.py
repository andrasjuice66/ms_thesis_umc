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
BATCH_SIZE  = 1
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
        log_probs: Log probabilities from model output [batch_size, n_bins]
        bin_centres: Age bin centers [n_bins]
    
    Returns:
        Predicted ages [batch_size]
    """
    probs = torch.exp(log_probs)  # Convert log probs to probs
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
    """Load test dataset from CSV."""
    print(f"Loading test data from {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} samples in test set")
    
    # Extract required columns
    required_cols = ['image_path', 'age']
    optional_cols = ['modality', 'sex']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    # Construct full paths
    file_paths = [os.path.join(data_root, path) for path in df['image_path']]
    ages = df['age'].tolist()
    
    # Optional columns
    modalities = df['modality'].tolist() if 'modality' in df.columns else None
    sexes = df['sex'].tolist() if 'sex' in df.columns else None
    
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
    
    # Base domain randomization config
    dr_config = {
        'device': DEVICE,
        'use_domain_randomization': True,
        'transform_probs': dr_config_from_file.get('transform_probs', {}),
        'output_shape': dr_config_from_file.get('output_shape', [160, 192, 160]),
        'use_tumor_simulation': use_tumor,
    }
    
    # Override tumor probability based on use_tumor parameter
    if 'transform_probs' in dr_config and 'tumor' in dr_config['transform_probs']:
        dr_config['transform_probs']['tumor'] = dr_config_from_file['transform_probs']['tumor'] if use_tumor else 0.0
    
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
            log_probs = outputs[0].squeeze()  # [batch_size, n_bins]
            
            # Convert to predicted ages
            pred_ages = predict_age_from_probabilities(log_probs, torch.tensor(bin_centres).to(DEVICE))
            
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


def main():
    """Main inference function."""
    print("Starting SFCN Inference Script")
    print(f"Device: {DEVICE}")
    print(f"Bin range: {BIN_RANGE}, step: {BIN_STEP}, bins: {N_BINS}")
    print("-" * 60)
    
    # Load configuration
    print(f"Loading configuration from: {CONFIG_PATH}")
    config = Config(CONFIG_PATH)
    
    # Initialize wandb
    if WANDB_ENABLED:
        wandb.login(key=WANDB_API)
        wandb.init(
            project="brainage-inference",
            name=f"sfcn_3regime_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'model_path': MODEL_PATH,
                'test_csv': TEST_CSV,
                'config_path': CONFIG_PATH,
                'bin_range': BIN_RANGE,
                'bin_step': BIN_STEP,
                'n_bins': N_BINS,
                'n_folds': N_FOLDS,
                'device': str(DEVICE),
                'batch_size': BATCH_SIZE,
                'domain_randomization_config': config.get('domain_randomization', {}),
            }
        )
    
    # Load model
    model = load_model(MODEL_PATH, DEVICE)
    
    # Load test data
    file_paths, ages, modalities, sexes = load_test_data(TEST_CSV, DATA_ROOT)
    
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
    all_results['domain_rand_mae_mean'] = dr_mean_mae
    all_results['domain_rand_mae_std'] = dr_std_mae
    
    print(f"\nDomain Randomization Results (10 folds):")
    print(f"  Mean MAE: {dr_mean_mae:.3f} ± {dr_std_mae:.3f}")
    
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
                all_results[f'domain_rand_mae_{mod}_mean'] = mod_mean
                all_results[f'domain_rand_mae_{mod}_std'] = mod_std
                print(f"  {mod.upper()} MAE: {mod_mean:.3f} ± {mod_std:.3f}")
    
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
    all_results['domain_rand_tumor_mae_mean'] = tumor_mean_mae
    all_results['domain_rand_tumor_mae_std'] = tumor_std_mae
    
    print(f"\nDomain Randomization + Tumor Results (10 folds):")
    print(f"  Mean MAE: {tumor_mean_mae:.3f} ± {tumor_std_mae:.3f}")
    
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
                all_results[f'domain_rand_tumor_mae_{mod}_mean'] = mod_mean
                all_results[f'domain_rand_tumor_mae_{mod}_std'] = mod_std
                print(f"  {mod.upper()} MAE: {mod_mean:.3f} ± {mod_std:.3f}")
    
    # ======================== SUMMARY ========================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"1. Normal Test MAE:           {all_results['normal_test_mae_overall']:.3f}")
    print(f"2. Domain Randomization MAE:  {dr_mean_mae:.3f} ± {dr_std_mae:.3f}")
    print(f"3. Domain Rand + Tumor MAE:   {tumor_mean_mae:.3f} ± {tumor_std_mae:.3f}")
    
    # Log all results to wandb
    if WANDB_ENABLED:
        wandb.log(all_results)
        
        # Create summary table
        summary_data = []
        summary_data.append(['Normal Test', all_results['normal_test_mae_overall'], 0])
        summary_data.append(['Domain Randomization', dr_mean_mae, dr_std_mae])
        summary_data.append(['Domain Rand + Tumor', tumor_mean_mae, tumor_std_mae])
        
        summary_table = wandb.Table(
            columns=['Regime', 'MAE', 'Std'],
            data=summary_data
        )
        wandb.log({'summary_table': summary_table})
        
        wandb.finish()
    
    # Save results to file
    results_file = OUT_DIR / f'sfcn_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()