"""
Evaluation script for brain age prediction models.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs.config import Config
from data.dataset import create_data_loaders
from models.sfcn import SFCN
from models.resnet3d import ResNet3D
from models.efficientnet3d import EfficientNet3D
from training.metrics import calculate_metrics
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate brain age prediction model")
    
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to evaluation CSV file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Path to output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    
    return parser.parse_args()


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        
    Returns:
        Dictionary of metrics and arrays of predictions and targets
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Get data
            inputs = batch["image"].to(device)
            targets = batch["age"].float().to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions and targets
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets


def plot_results(predictions, targets, output_dir):
    """
    Plot evaluation results.
    
    Args:
        predictions: Predicted ages
        targets: True ages
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scatter plot of predictions vs. targets
    plt.figure(figsize=(10, 10))
    
    # Plot identity line
    min_val = min(np.min(predictions), np.min(targets))
    max_val = max(np.max(predictions), np.max(targets))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Plot predictions vs. targets
    plt.scatter(targets, predictions, alpha=0.5)
    
    # Add labels and title
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title('Predictions vs. Targets')
    
    # Add correlation coefficient
    correlation = np.corrcoef(predictions, targets)[0, 1]
    plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes)
    
    # Save figure
    plt.savefig(output_dir / "predictions_vs_targets.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Histogram of prediction errors
    errors = predictions - targets
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error (years)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig(output_dir / "prediction_errors.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Age-specific MAE
    age_bins = np.arange(0, 101, 10)
    bin_indices = np.digitize(targets, age_bins) - 1
    bin_maes = []
    bin_counts = []
    
    for i in range(len(age_bins) - 1):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_mae = mean_absolute_error(targets[bin_mask], predictions[bin_mask])
            bin_maes.append(bin_mae)
            bin_counts.append(np.sum(bin_mask))
        else:
            bin_maes.append(0)
            bin_counts.append(0)
    
    bin_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins) - 1)]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(bin_labels, bin_maes)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'n={count}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Age Range')
    plt.ylabel('MAE (years)')
    plt.title('Age-Specific MAE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "age_specific_mae.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = args.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = setup_logger("evaluate", output_dir / "evaluate.log")
    logger.info(f"Arguments: {args}")
    
    # Load model checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model configuration
    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get("type", "sfcn")
    
    # Create model
    logger.info(f"Creating model: {model_type}")
    if model_type == "sfcn":
        model = SFCN(**model_config)
    elif model_type == "resnet3d":
        model = ResNet3D(**model_config)
    elif model_type == "efficientnet3d":
        model = EfficientNet3D(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Create data loader
    logger.info("Creating data loader...")
    data_loader = create_data_loaders(
        train_csv=args.data_csv,  # Use the same CSV for both to create only one loader
        val_csv=args.data_csv,
        image_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        domain_randomization_params=None  # No randomization for evaluation
    )[1]  # Get only the validation loader
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics, predictions, targets = evaluate_model(model, data_loader, device)
    
    # Log metrics
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        "true_age": targets.flatten(),
        "predicted_age": predictions.flatten(),
        "error": predictions.flatten() - targets.flatten()
    })
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Plot results
    logger.info("Plotting results...")
    plot_results(predictions.flatten(), targets.flatten(), output_dir)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 