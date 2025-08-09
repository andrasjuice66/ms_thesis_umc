#!/usr/bin/env python
"""
Evaluation script for original SFCN paper weights using the standardized testing regime.

This script uses the original SFCN model architecture and weight loading approach
from the inference_sfcn.py script, but applies the comprehensive testing regime
from the evaluate.py script, including:
- Overall metrics (MAE, MSE, R², Correlation)
- Per-modality metrics 
- Per-sex metrics
- Per-headmotion metrics
- W&B logging with detailed tables

The SFCN model expects age-bin classification output and converts to age predictions
using the same bin centers and prediction logic as the original script.
"""

import os, sys, json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

# Set multiprocessing start method BEFORE other imports
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from brain_age_pred.dataset.dataset import BADataset
from brain_age_pred.utils.utils import read_csv
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.configs.config import Config


# Bin settings - MUST match training
BIN_RANGE = (42, 82)   # inclusive range used to build bins
BIN_STEP = 1
bin_centres = np.arange(BIN_RANGE[0] + BIN_STEP / 2,
                        BIN_RANGE[1] + BIN_STEP / 2,
                        BIN_STEP, dtype=np.float32)
N_BINS = len(bin_centres)


class SFCN(nn.Module):
    """Original SFCN model architecture from the paper"""
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


def load_test_data_with_headmotion(csv_path, data_root):
    """Load test dataset from CSV with headmotion support and filter by age range."""
    print(f"Loading test data from {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} samples in original test set")
    
    # Extract required columns
    required_cols = ['image_path', 'age']
    optional_cols = ['modality', 'sex', 'headmotion']
    
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
    
    # Filter out missing files - collect valid data aligned with file existence
    valid_file_paths = []
    valid_ages = []
    valid_modalities = []
    valid_sexes = []
    valid_headmotions = []
    
    missing_count = 0
    for idx, row in df_filtered.iterrows():
        file_path = os.path.join(data_root, row['image_path'])
        if os.path.exists(file_path):
            valid_file_paths.append(file_path)
            valid_ages.append(row['age'])
            valid_modalities.append(row.get('modality', 'N/A'))
            valid_sexes.append(row.get('sex', 'N/A'))
            valid_headmotions.append(row.get('headmotion', 'N/A'))
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing files
                print(f"  Missing: {file_path}")
    
    if missing_count > 0:
        print(f"Warning: {missing_count} files not found - filtered out from evaluation")
        print(f"Remaining samples after file filtering: {len(valid_file_paths)}")
    
    if len(valid_file_paths) == 0:
        raise ValueError("No valid files found after filtering")
    
    # Convert modality lists to None if all are 'N/A' to maintain compatibility
    modalities = valid_modalities if any(m != 'N/A' for m in valid_modalities) else None
    sexes = valid_sexes if any(s != 'N/A' for s in valid_sexes) else None
    headmotions = valid_headmotions if any(h != 'N/A' for h in valid_headmotions) else None
    
    # Print modality distribution if available
    if modalities:
        modality_series = pd.Series(modalities)
        modality_counts = modality_series.value_counts()
        print("Modality distribution:")
        for mod, count in modality_counts.items():
            print(f"  {mod}: {count} samples")
    
    return valid_file_paths, valid_ages, modalities, sexes, headmotions


def load_sfcn_model(model_path, device):
    """Load SFCN model from checkpoint using original loading approach with enhanced error handling"""
    print(f"Loading SFCN model from {model_path}")
    
    # Initialize model
    model = SFCN(output_dim=N_BINS, dropout=True)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Try multiple loading approaches for problematic checkpoints
    checkpoint = None
    loading_error = None
    
    # Method 1: Standard loading
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("✓ Loaded using standard method")
    except Exception as e:
        loading_error = str(e)
        print(f"✗ Standard loading failed: {e}")
        
        # Method 2: Try with different pickle settings
        try:
            import pickle
            checkpoint = torch.load(
                model_path, 
                map_location=device, 
                weights_only=False, 
                pickle_module=pickle
            )
            print("✓ Loaded using custom pickle module")
        except Exception as e2:
            print(f"✗ Custom pickle loading failed: {e2}")
            
            # Method 3: Try loading with BOM handling
            try:
                with open(model_path, 'rb') as f:
                    data = f.read()
                    
                # Check for UTF-8 BOM and remove if present
                if data.startswith(b'\xef\xbb\xbf'):
                    print("Detected UTF-8 BOM, removing...")
                    data = data[3:]
                elif data.startswith(b'\xef'):
                    print("Detected potential BOM corruption, removing first byte...")
                    data = data[1:]
                
                # Save to temporary file and reload
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_f:
                    tmp_f.write(data)
                    tmp_path = tmp_f.name
                
                checkpoint = torch.load(tmp_path, map_location=device, weights_only=False)
                os.unlink(tmp_path)  # Clean up
                print("✓ Loaded after BOM/corruption removal")
                
            except Exception as e3:
                print(f"✗ BOM removal loading failed: {e3}")
                
                # Method 4: Try pure Python pickle with encoding handling
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        # Try to skip problematic bytes
                        f.seek(1)  # Skip first byte
                        checkpoint = pickle.load(f)
                    print("✓ Loaded using pure pickle with byte skip")
                except Exception as e4:
                    print(f"✗ Pure pickle with skip failed: {e4}")
                    
                    # Method 5: Last resort - try different encodings
                    try:
                        # This is a very specific fix for certain corrupted pickle files
                        with open(model_path, 'rb') as f:
                            data = f.read()
                        
                        # Try to find the actual pickle data start
                        for i in range(min(10, len(data))):
                            try:
                                import io
                                stream = io.BytesIO(data[i:])
                                checkpoint = torch.load(stream, map_location=device, weights_only=False)
                                print(f"✓ Loaded by skipping {i} bytes at start")
                                break
                            except:
                                continue
                        
                        if checkpoint is None:
                            raise Exception("Could not find valid pickle data")
                            
                    except Exception as e5:
                        print(f"✗ All methods failed: {e5}")
                        raise RuntimeError(f"All loading methods failed. Original error: {loading_error}")
    
    if checkpoint is None:
        raise RuntimeError(f"Failed to load checkpoint. Error: {loading_error}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Using 'state_dict' key from checkpoint")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Using 'model_state_dict' key from checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
        print("Checkpoint is a direct state_dict")
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    print(f"Loading {len(new_state_dict)} parameters into model")
    
    # Load with strict=False to handle potential parameter mismatches
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    model.to(device)
    model.eval()
    
    print(f"SFCN model loaded successfully on {device}")
    return model


def evaluate_model_sfcn(model, test_loader, device, headmotions=None):
    """Runs the evaluation loop for SFCN model with comprehensive metadata tracking."""
    model.eval()
    all_preds = []
    all_ages = []
    all_modalities = []
    all_sexes = []
    all_headmotions = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Get image and age
            image = batch['image'].to(device)
            age = batch['age'].to(device)
            
            # Get modality and sex if available
            modality = batch.get('modality', [None] * len(age))
            sex = batch.get('sex', [None] * len(age))
            
            # Model inference
            outputs = model(image)
            log_probs = outputs[0]  # [batch_size, n_bins, 1, 1, 1]
            
            # Remove spatial dimensions but keep batch dimension
            log_probs = log_probs.squeeze(-1).squeeze(-1).squeeze(-1)  # [batch_size, n_bins]
            
            # Convert to predicted ages
            pred_ages = predict_age_from_probabilities(log_probs, torch.tensor(bin_centres).to(device))
            
            # Ensure pred_ages is always a tensor with batch dimension
            if len(pred_ages.shape) == 0:  # scalar
                pred_ages = pred_ages.unsqueeze(0)
            
            # Store results
            all_preds.extend(pred_ages.cpu().numpy())
            all_ages.extend(age.cpu().numpy())
            all_modalities.extend(modality)
            all_sexes.extend(sex)
            
            # Add headmotion data if available
            if headmotions is not None:
                batch_size = len(age)
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
    parser = argparse.ArgumentParser(description="SFCN Original Weights Evaluation")
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
    logger = setup_logger("sfcn_evaluation", log_file=log_dir / "sfcn_eval.log")
    
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")
    logger.info(f"Bin range: {BIN_RANGE}, step: {BIN_STEP}, bins: {N_BINS}")
    
    # W&B Setup
    use_wandb = cfg.get("wandb.use_wandb", True)
    experiment_name = cfg.get("wandb.experiment_name")
    if not experiment_name:
        experiment_name = f'sfcn_original_evaluation_{timestamp}'
    
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
        
        # Skip non-SFCN models
        if model_info.get("params", {}).get("type") != "sfcn_original":
            logger.info(f"Skipping non-SFCN model: {model_name}")
            continue
            
        logger.info(f"--- Evaluating SFCN model: {model_name} ---")
        evaluation_results[model_name] = {}
        
        # Load SFCN model
        checkpoint_path = model_info["checkpoint"]
        try:
            model = load_sfcn_model(checkpoint_path, device)
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
            
            # Load test data with filtering for SFCN age range
            file_paths, ages, modalities, sexes, headmotions = load_test_data_with_headmotion(test_csv, data_dir)
            
            # Create dataset (no transforms for SFCN original - raw data)
            test_dataset = BADataset(
                file_paths=file_paths,
                age_labels=ages,
                modalities=modalities,
                sexes=sexes,
                transform=None,  # No augmentation for original weights
                mode='test'
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.get("batch_size", 8),
                shuffle=False,
                num_workers=cfg.get("num_workers", 4),
                pin_memory=True
            )
            
            # Evaluate and calculate metrics
            preds, true_ages, mods, sxs, hmotions = evaluate_model_sfcn(
                model, test_loader, device, headmotions
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
    results_file = log_dir / "sfcn_evaluation_summary.json"
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    logger.info(f"Full evaluation results saved to {results_file}")
    
    if use_wandb:
        wandb.finish()
        logger.info("W&B session finished")


if __name__ == "__main__":
    main()