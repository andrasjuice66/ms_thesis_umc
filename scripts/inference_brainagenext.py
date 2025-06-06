#!/usr/bin/env python
"""
Inference script for BrainAgeNeXt model that loads a test dataloader with targets
and evaluates performance using comprehensive metrics.
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.utils.utils import read_csv, load_checkpoint
from brain_age_pred.training.metrics import calculate_metrics
from brain_age_pred.models.brainagenext import BrainAgeNeXt

def create_test_dataloader(csv_path, data_dir, batch_size=8, num_workers=4):
    """Create a DataLoader for test data with targets"""
    # Read CSV file to get file paths and age labels
    file_paths, ages, sample_weights, sexes, modalities = read_csv(csv_path, data_dir)
    
    print(f"Loaded {len(file_paths)} samples from {csv_path}")
    print(f"Age range: {min(ages):.2f} - {max(ages):.2f}, mean: {np.mean(ages):.2f}")
    
    # Create dataset
    test_dataset = BADataset(
        file_paths=file_paths,
        age_labels=ages,
        sexes=sexes,
        modalities=modalities,
        transform=None,
        mode="test",
        cache_size=0,
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader, file_paths, ages, sexes, modalities

def inference_with_dataloader():
    # Configure paths and parameters
    model_path = '/home/ajoos/model_files/brainagenext_best.pt'  # Update with actual model path
    test_csv_path = '/home/ajoos/brain_age_pred/data/labels/test.csv'
    data_dir = '/scratch-shared/ajoos/'
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running on device: {device}")
    
    # Create test dataloader
    test_loader, file_paths, true_ages, sexes, modalities = create_test_dataloader(
        test_csv_path, data_dir, batch_size=batch_size
    )
    
    print(f"Loaded test dataloader with {len(test_loader)} batches ({len(file_paths)} samples)")
    
    # Load model
    model = BrainAgeNeXt(
        in_channels=1,
        dropout_rate=0.0,
        model_id='B',
        kernel_size=3,
        deep_supervision=True,
        feature_size=512,
        hidden_size=64
    )
    
    print(f"Loading model from {model_path}")
    # Handle loading with or without DataParallel
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if the state dict was saved with DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        # Model was saved with DataParallel, wrap the model
        model = torch.nn.DataParallel(model)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Evaluation loop
    all_predictions = []
    all_targets = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get inputs and targets
            inputs = batch["image"].to(device)
            targets = batch["age"].to(device)
            
            # Forward pass
            predictions = model(inputs)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}/{len(test_loader)}")
    
    # Convert to numpy arrays for analysis
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate comprehensive metrics using the project's metrics function
    metrics = calculate_metrics(predictions, targets, modalities, sexes)
    
    # Print detailed metrics
    print("\nEvaluation Results:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Create and save visualizations
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title(f'Brain Age Prediction Results (MAE={metrics["mae"]:.2f}, R²={metrics["r2"]:.2f})')
    plt.grid(True)
    plt.savefig('brainagenext_prediction_results.png')
    
    # Create brain age delta plot
    plt.figure(figsize=(10, 6))
    brain_age_delta = predictions - targets
    plt.hist(brain_age_delta, bins=30)
    plt.xlabel('Brain Age Delta (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title(f'Brain Age Delta Distribution (Mean={metrics["brain_age_delta"]:.2f})')
    plt.grid(True)
    plt.savefig('brainagenext_delta_distribution.png')
    
    # If there are multiple modalities, create modality-specific plots
    if modalities is not None and len(np.unique(modalities)) > 1:
        plt.figure(figsize=(12, 8))
        unique_modalities = np.unique(modalities)
        for i, modality in enumerate(unique_modalities):
            mask = np.array(modalities) == modality
            plt.scatter(targets[mask], predictions[mask], alpha=0.5, label=f'{modality} (MAE={metrics.get(f"{modality}_mae", 0):.2f})')
        
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--')
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        plt.title('Brain Age Prediction by Modality')
        plt.legend()
        plt.grid(True)
        plt.savefig('brainagenext_by_modality.png')
    
    # If there are sex values, create sex-specific plots
    if sexes is not None and len(np.unique(sexes)) > 1:
        plt.figure(figsize=(12, 8))
        unique_sexes = np.unique(sexes)
        for i, sex in enumerate(unique_sexes):
            mask = np.array(sexes) == sex
            plt.scatter(targets[mask], predictions[mask], alpha=0.5, label=f'{sex} (MAE={metrics.get(f"{sex}_mae", 0):.2f})')
        
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--')
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        plt.title('Brain Age Prediction by Sex')
        plt.legend()
        plt.grid(True)
        plt.savefig('brainagenext_by_sex.png')
    
    # Save age-specific MAE bar chart
    age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
    age_specific_mae = {}
    for i in range(len(age_bins) - 1):
        bin_start = age_bins[i]
        bin_end = age_bins[i+1]
        metric_key = f"mae_{bin_start}_{bin_end}"
        if metric_key in metrics:
            age_specific_mae[f"{bin_start}-{bin_end}"] = metrics[metric_key]
    
    if age_specific_mae:
        plt.figure(figsize=(10, 6))
        plt.bar(age_specific_mae.keys(), age_specific_mae.values())
        plt.xlabel('Age Range')
        plt.ylabel('MAE')
        plt.title('Age-Specific Mean Absolute Error')
        plt.grid(axis='y')
        plt.savefig('brainagenext_age_specific_mae.png')
    
    # Save all plots
    plt.close('all')
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'file_path': file_paths,
        'true_age': targets,
        'predicted_age': predictions,
        'brain_age_delta': predictions - targets,
    })
    if sexes is not None:
        results_df['sex'] = sexes
    if modalities is not None:
        results_df['modality'] = modalities
    
    results_df.to_csv('brainagenext_prediction_results.csv', index=False)
    print("Results saved to brainagenext_prediction_results.csv")
    
    # Return metrics for further analysis if needed
    return metrics, results_df

if __name__ == "__main__":
    inference_with_dataloader() 