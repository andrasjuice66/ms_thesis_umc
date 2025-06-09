#!/usr/bin/env python
"""
Inference script for BrainAgeNeXt model with 3-fold evaluation regime and 5-model ensemble:
1. Normal test evaluation
2. Domain randomized test evaluation (10 folds, averaged)
3. Domain randomized + tumor simulation test evaluation (10 folds, averaged)

Uses 5-model ensemble with median predictions and brain age correction as in original script.
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
import wandb
from datetime import datetime
from torch.utils.data import DataLoader
import torchio
from monai.transforms import Compose, ScaleIntensityd, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd, MapTransform
from monai.data import CacheDataset

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer
from brain_age_pred.utils.utils import read_csv, load_checkpoint
from brain_age_pred.training.metrics import calculate_metrics

# Import the nnunet_mednext for the original architecture
from brain_age_pred.models.create_mednext_encoder_v1 import create_mednext_encoder_v1


class LoadNumpyArrayd(MapTransform):
    """Custom MONAI transform to load numpy arrays (.npy files)"""
    def __init__(self, keys, ensure_channel_first=True):
        super().__init__(keys)
        self.ensure_channel_first = ensure_channel_first

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], str):
                # Load numpy array
                array = np.load(d[key]).astype(np.float32)
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


def prepare_monai_transforms():
    """Prepare MONAI transforms for numpy arrays"""
    x, y, z = (160, 192, 160)
    p = 1.0
    monai_transforms = [
        LoadNumpyArrayd(keys=["image"], ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=(p, p, p)),
        CropForegroundd(keys=["image"], allow_smaller=True, source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(x, y, z)),
        CenterSpatialCropd(keys=["image"], roi_size=(x, y, z))
    ]
    val_torchio_transforms = torchio.transforms.Compose(
        [torchio.transforms.ZNormalization(masking_method=lambda x: x > 0, keys=["image"], include=['image'])]
    )
    return Compose(monai_transforms + [val_torchio_transforms])


def create_monai_dataloader(csv_path, data_dir, batch_size=1, num_workers=4):
    """Create dataloader using MONAI transforms for numpy arrays"""
    # Read CSV and prepare data dicts
    df = pd.read_csv(csv_path)
    df.dropna(subset=['image_path'], inplace=True)
    df.dropna(subset=['age'], inplace=True)
    
    # Adjust paths to be relative to data_dir
    data_dicts = []
    for _, row in df.iterrows():
        image_path = os.path.join(data_dir, row['image_path']) if not os.path.isabs(row['image_path']) else row['image_path']
        data_dicts.append({'image': image_path, 'label': row['age']})
    
    # Create transforms and dataset
    transforms = prepare_monai_transforms()
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.2, num_workers=num_workers)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, 
                           pin_memory=torch.cuda.is_available())
    
    return dataloader, df


def create_test_dataloader(csv_path, data_dir, transform=None, batch_size=8, num_workers=4):
    """Create a DataLoader for test data with targets (for domain randomization)"""
    if transform is None:
        # Use MONAI transforms for normal evaluation
        return create_monai_dataloader(csv_path, data_dir, batch_size, num_workers)
    
    # Use custom transforms for domain randomization
    file_paths, ages, sample_weights, sexes, modalities = read_csv(csv_path, data_dir)
    
    test_dataset = BADataset(
        file_paths=file_paths,
        age_labels=ages,
        sexes=sexes,
        modalities=modalities,
        transform=transform,
        mode="test",
        cache_size=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader, file_paths, ages, sexes, modalities


def create_eval_transforms(device, use_domain_rand=False, use_tumor=False):
    """Create evaluation-specific transforms"""
    if not use_domain_rand:
        return None
    
    # Default domain randomization config for evaluation
    eval_rand_cfg = {
        "use_domain_randomization": True,
        "transform_probs": {
            "flip": 0.5,
            "affine": 0.8,
            "contrast": 0.6,
            "gamma": 0.5,
            "blur": 0.4,
            "bias": 0.5,
            "scale_int": 0.4,
            "shift_int": 0.4,
            "hist_shift": 0.3,
            "noise": 0.4,
            "rician": 0.3,
            "gibbs": 0.3,
            "resolution": 0.5,
            "coarse_do": 0.3,
            "crop": 1.0,
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
    
    eval_transform = DomainRandomizer(
        device=device,
        use_tumor_simulation=use_tumor,
        tumor_config=eval_tumor_cfg,
        **eval_rand_cfg,
    )
    
    return eval_transform


def initialize_model(device):
    """Initialize a single model"""
    torch.cuda.empty_cache()
    return MedNeXtEncReg().to(device)


def run_predictions_single_model(model_path, dataloader, device):
    """Run predictions with a single model"""
    model = initialize_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            images = batch_data['image'].to(device)
            
            # Handle both MONAI dataloader ('label') and BADataset ('age') keys
            if 'label' in batch_data:
                labels = batch_data['label'].to(device)
            elif 'age' in batch_data:
                labels = batch_data['age'].to(device)
            else:
                raise KeyError("Neither 'label' nor 'age' key found in batch data")
            
            pred = model(images)
            # Convert to numpy and ensure it's at least 1D for extending the list
            pred_np = pred.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Handle both scalar (0-d) and vector outputs
            if pred_np.ndim == 0:
                predictions.append(pred_np.item())
            else:
                predictions.extend(pred_np)
                
            if labels_np.ndim == 0:
                targets.append(labels_np.item())
            else:
                targets.extend(labels_np)
    
    del model
    torch.cuda.empty_cache()
    return np.array(predictions), np.array(targets)


def run_ensemble_predictions(model_paths, dataloader, device):
    """Run predictions with 5-model ensemble using median"""
    all_predictions = []
    targets = None
    
    for i, model_path in enumerate(model_paths):
        print(f"Running predictions with model {i+1}/5: {os.path.basename(model_path)}")
        pred, tgt = run_predictions_single_model(model_path, dataloader, device)
        all_predictions.append(pred)
        if targets is None:
            targets = tgt
    
    # Calculate median across models
    ensemble_predictions = np.median(np.stack(all_predictions), axis=0)
    return ensemble_predictions, targets


def apply_brain_age_correction(predicted_ages, chronological_ages):
    """Apply brain age correction as in original script"""
    # BA_corr = np.where(CA > 18, BA + (CA * 0.062) - 2.96, BA)
    corrected_ages = np.where(
        chronological_ages > 18, 
        predicted_ages + (chronological_ages * 0.062) - 2.96, 
        predicted_ages
    )
    return corrected_ages


def run_single_evaluation_ensemble(model_paths, test_loader, device):
    """Run a single evaluation with ensemble on the test loader"""
    predictions, targets = run_ensemble_predictions(model_paths, test_loader, device)
    
    # Apply brain age correction
    corrected_predictions = apply_brain_age_correction(predictions, targets)
    
    return corrected_predictions, targets


def run_multi_fold_evaluation_ensemble(model_paths, csv_path, data_dir, device, transform, n_folds, eval_name, batch_size=8):
    """Run evaluation multiple times with different augmentations and average results"""
    print(f"Running {n_folds}-fold {eval_name} evaluation with ensemble...")
    
    all_metrics = []
    
    for fold in range(n_folds):
        print(f"{eval_name} evaluation fold {fold+1}/{n_folds}")
        
        # Create test dataloader with transform
        if transform is None:
            test_loader, df = create_monai_dataloader(csv_path, data_dir, batch_size)
            # Extract modalities and sexes from df if available
            modalities = df.get('modality', [None] * len(df)).values
            sexes = df.get('sex', [None] * len(df)).values
        else:
            test_loader, file_paths, ages, sexes, modalities = create_test_dataloader(
                csv_path, data_dir, transform=transform, batch_size=batch_size
            )
        
        # Run evaluation
        predictions, targets = run_ensemble_predictions(model_paths, test_loader, device)
        corrected_predictions = apply_brain_age_correction(predictions, targets)
        
        # Calculate metrics
        metrics = calculate_metrics(corrected_predictions, targets, modalities, sexes)
        all_metrics.append(metrics)
    
    # Average metrics across folds
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)
    
    print(f"{eval_name} evaluation results (averaged over {n_folds} folds):")
    print(f"MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['mae_std']:.4f}")
    print(f"MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['mse_std']:.4f}")
    print(f"R²: {avg_metrics['r2']:.4f} ± {avg_metrics['r2_std']:.4f}")
    
    return avg_metrics


def inference_with_3fold_evaluation():
    # Configure paths and parameters
    model_dir = '/home/ajoos/model_files/'
    model_paths = [os.path.join(model_dir, f'BrainAge_{i}.pth') for i in range(1, 6)]
    test_csv_path = '/home/ajoos/brain_age_pred/data/labels/test.csv'
    data_dir = '/scratch-shared/ajoos/'
    batch_size = 1  # Use batch_size=1 as in original script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Verify model files exist
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
    
    # Initialize W&B
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"brainagenext_ensemble_{timestamp}"
    
    WANDB_API = '2abdb867a9244072f2237704a3cacc77fa548dd8'
    wandb.login(key=WANDB_API)
    wandb.init(
        project="brainage-inference",
        name=experiment_name,
        config={
            "model": "BrainAgeNeXt_Ensemble",
            "model_paths": model_paths,
            "test_csv": test_csv_path,
            "batch_size": batch_size,
            "device": str(device),
            "evaluation_type": "3fold_ensemble_inference",
            "num_models": len(model_paths),
            "ensemble_method": "median"
        },
        reinit=True,
    )
    
    print(f"Running on device: {device}")
    print(f"W&B experiment: {experiment_name}")
    print(f"Using {len(model_paths)} models for ensemble")
    
    try:
        # Read dataset info
        df = pd.read_csv(test_csv_path)
        print(f"Loaded {len(df)} samples from {test_csv_path}")
        print(f"Age range: {df['age'].min():.2f} - {df['age'].max():.2f}, mean: {df['age'].mean():.2f}")
        
        # Log dataset info to W&B
        wandb.log({
            "dataset/num_samples": len(df),
            "dataset/age_min": df['age'].min(),
            "dataset/age_max": df['age'].max(),
            "dataset/age_mean": df['age'].mean(),
            "dataset/age_std": df['age'].std(),
        })
        
        print("Starting 3-fold evaluation with ensemble...")
        
        # 1. Normal test evaluation with ensemble
        print("=== 1/3: Normal test evaluation (ensemble) ===")
        normal_test_loader, normal_df = create_monai_dataloader(test_csv_path, data_dir, batch_size)
        normal_predictions, normal_targets = run_ensemble_predictions(model_paths, normal_test_loader, device)
        normal_corrected_predictions = apply_brain_age_correction(normal_predictions, normal_targets)
        
        # Get modalities and sexes if available
        modalities = normal_df.get('modality', [None] * len(normal_df)).values
        sexes = normal_df.get('sex', [None] * len(normal_df)).values
        
        normal_metrics = calculate_metrics(normal_corrected_predictions, normal_targets, modalities, sexes)
        print(f"Normal test results: MAE={normal_metrics['mae']:.4f}, R²={normal_metrics['r2']:.4f}")
        
        # Log normal test results to W&B
        wandb.log({f"test/{k}": v for k, v in normal_metrics.items()})
        
        # 2. Domain randomized test evaluation (10 folds)
        print("=== 2/3: Domain randomized test evaluation (ensemble) ===")
        dom_rand_transform = create_eval_transforms(device, use_domain_rand=True, use_tumor=False)
        dom_rand_metrics = run_multi_fold_evaluation_ensemble(
            model_paths, test_csv_path, data_dir, device, dom_rand_transform, 
            n_folds=10, eval_name="domain_randomized", batch_size=batch_size
        )
        
        # Log domain randomized results to W&B
        wandb.log({f"test_dom_rand/{k}": v for k, v in dom_rand_metrics.items()})
        
        # 3. Domain randomized + tumor simulation test evaluation (10 folds)
        print("=== 3/3: Domain randomized + tumor simulation test evaluation (ensemble) ===")
        dom_rand_tumor_transform = create_eval_transforms(device, use_domain_rand=True, use_tumor=True)
        dom_rand_tumor_metrics = run_multi_fold_evaluation_ensemble(
            model_paths, test_csv_path, data_dir, device, dom_rand_tumor_transform, 
            n_folds=10, eval_name="domain_rand_tumor", batch_size=batch_size
        )
        
        # Log domain randomized + tumor results to W&B
        wandb.log({f"test_dom_rand_tumor/{k}": v for k, v in dom_rand_tumor_metrics.items()})
        
        # Log summary comparison to W&B
        wandb.log({
            "evaluation_summary/normal_mae": normal_metrics["mae"],
            "evaluation_summary/dom_rand_mae": dom_rand_metrics["mae"],
            "evaluation_summary/dom_rand_tumor_mae": dom_rand_tumor_metrics["mae"],
            "evaluation_summary/dom_rand_mae_std": dom_rand_metrics["mae_std"],
            "evaluation_summary/dom_rand_tumor_mae_std": dom_rand_tumor_metrics["mae_std"],
            "evaluation_summary/normal_r2": normal_metrics["r2"],
            "evaluation_summary/dom_rand_r2": dom_rand_metrics["r2"],
            "evaluation_summary/dom_rand_tumor_r2": dom_rand_tumor_metrics["r2"],
        })
        
        # Save evaluation results
        eval_results = {
            "normal": normal_metrics,
            "domain_randomized": dom_rand_metrics,
            "domain_rand_tumor": dom_rand_tumor_metrics,
        }
        
        with open('brainagenext_ensemble_3fold_evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print("=== Evaluation Summary ===")
        print(f"Normal test MAE: {normal_metrics['mae']:.4f}")
        print(f"Domain rand test MAE: {dom_rand_metrics['mae']:.4f} ± {dom_rand_metrics['mae_std']:.4f}")
        print(f"Domain rand + tumor test MAE: {dom_rand_tumor_metrics['mae']:.4f} ± {dom_rand_tumor_metrics['mae_std']:.4f}")
        
        # Create visualization comparing all three evaluations
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Normal evaluation
        plt.subplot(1, 3, 1)
        plt.scatter(normal_targets, normal_corrected_predictions, alpha=0.5)
        plt.plot([min(normal_targets), max(normal_targets)], [min(normal_targets), max(normal_targets)], 'r--')
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age (Corrected)')
        plt.title(f'Normal Test (Ensemble)\n(MAE={normal_metrics["mae"]:.2f}, R²={normal_metrics["r2"]:.2f})')
        plt.grid(True)
        
        # Plot 2: MAE comparison
        plt.subplot(1, 3, 2)
        mae_values = [normal_metrics['mae'], dom_rand_metrics['mae'], dom_rand_tumor_metrics['mae']]
        mae_stds = [0, dom_rand_metrics['mae_std'], dom_rand_tumor_metrics['mae_std']]
        labels = ['Normal', 'Domain Rand', 'Dom Rand + Tumor']
        plt.bar(labels, mae_values, yerr=mae_stds, capsize=5)
        plt.ylabel('MAE')
        plt.title('MAE Comparison (Ensemble)')
        plt.grid(axis='y')
        
        # Plot 3: R² comparison
        plt.subplot(1, 3, 3)
        r2_values = [normal_metrics['r2'], dom_rand_metrics['r2'], dom_rand_tumor_metrics['r2']]
        r2_stds = [0, dom_rand_metrics['r2_std'], dom_rand_tumor_metrics['r2_std']]
        plt.bar(labels, r2_values, yerr=r2_stds, capsize=5)
        plt.ylabel('R²')
        plt.title('R² Comparison (Ensemble)')
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig('brainagenext_ensemble_3fold_evaluation_comparison.png', dpi=300, bbox_inches='tight')
        
        # Log the plot to W&B
        wandb.log({"evaluation_plots": wandb.Image('brainagenext_ensemble_3fold_evaluation_comparison.png')})
        plt.close()
        
        # Save detailed normal test results with brain age delta
        brain_age_delta = normal_corrected_predictions - normal_targets
        
        results_df = pd.DataFrame({
            'file_path': normal_df['image_path'].values,
            'true_age': normal_targets,
            'predicted_age': normal_predictions,  # Raw predictions
            'predicted_age_corrected': normal_corrected_predictions,  # Corrected predictions
            'brain_age_delta': brain_age_delta,
        })
        
        if 'sex' in normal_df.columns:
            results_df['sex'] = normal_df['sex'].values
        if 'modality' in normal_df.columns:
            results_df['modality'] = normal_df['modality'].values
        
        results_df.to_csv('brainagenext_ensemble_normal_test_results.csv', index=False)
        print("Results saved to brainagenext_ensemble_normal_test_results.csv and brainagenext_ensemble_3fold_evaluation_results.json")
        
        return eval_results
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        wandb.finish()


# Keep the original function for backward compatibility
def inference_with_dataloader():
    return inference_with_3fold_evaluation()


if __name__ == "__main__":
    inference_with_3fold_evaluation() 