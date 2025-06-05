#!/usr/bin/env python
"""
Script to load and evaluate brain age prediction models on validation and test sets.
Supports both SFCN and BrainAgeNeXt models, using a dedicated evaluation config file.

Usage:
    python evaluate.py --config <eval_config_path>
"""
import os, sys, argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.models.sfcn import SFCN
from brain_age_pred.models.sfcn_class import SFCNClass
from brain_age_pred.models.brainagenext import BrainAgeNeXt
from brain_age_pred.training.metrics import calculate_metrics
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate brain age models using a dedicated config file")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config file")
    return parser.parse_args()


def create_data_loader(file_paths, age_labels, sexes, modalities, batch_size, num_workers=4):
    """Create a DataLoader for evaluation"""
    dataset = BADataset(
        file_paths=file_paths,
        age_labels=age_labels,
        sexes=sexes,
        modalities=modalities,
        transform=None,
        mode="eval",
        cache_size=0,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


def load_model(model_config, checkpoint_path, device, logger):
    """Load the specified model type with the given checkpoint"""
    model_type = model_config.get("type")
    age_min = model_config.get("age_min", 20)
    age_max = model_config.get("age_max", 80)
    
    if model_type == "sfcn":
        model = SFCN(
            in_channels=model_config.get("in_channels", 1),
            dropout_rate=model_config.get("dropout_rate", 0.5),
            age_min=age_min,
            age_max=age_max,
        ).to(device)
    elif model_type == "sfcn_class":
        model = SFCNClass(
            in_channels=model_config.get("in_channels", 1),
            dropout_rate=model_config.get("dropout_rate", 0.5),
            channels=model_config.get("channels", (32, 64, 128, 256, 256, 64)),
            age_min=age_min,
            age_max=age_max,
        ).to(device)
    elif model_type == "brainagenext":
        model = BrainAgeNeXt(
            in_channels=model_config.get("in_channels", 1),
            dropout_rate=model_config.get("dropout_rate", 0.0),
            model_id=model_config.get("model_id", 'B'),
            kernel_size=model_config.get("kernel_size", 3),
            deep_supervision=model_config.get("deep_supervision", True),
            feature_size=model_config.get("feature_size", 512),
            hidden_size=model_config.get("hidden_size", 64)
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint_info = load_checkpoint(model, checkpoint_path, device, logger)
    if checkpoint_info:
        logger.info(f"Loaded checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
    
    return model


def evaluate_model(model, data_loader, device, model_type):
    """Evaluate the model on the given data loader"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_sexes = []
    all_modalities = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)
            sexes = batch["sex"]
            modalities = batch["modality"]
            
            if model_type == "sfcn_class":
                # For classification model, get log probabilities and convert to expected age
                log_probs = model(inputs)
                predictions = model.expected_age(log_probs)
            else:
                # For regression models, get predictions directly
                predictions = model(inputs)
            
            # Store batch results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_sexes.extend(sexes)
            all_modalities.extend(modalities)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    sexes = np.array(all_sexes)
    modalities = np.array(all_modalities)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets, modalities, sexes)
    
    return metrics, predictions, targets, sexes, modalities


def save_predictions(file_paths, predictions, targets, sexes, modalities, output_path):
    """Save predictions to a CSV file"""
    df = pd.DataFrame({
        'file_path': file_paths,
        'true_age': targets,
        'predicted_age': predictions,
        'sex': sexes,
        'modality': modalities,
        'brain_age_delta': predictions - targets
    })
    df.to_csv(output_path, index=False)
    return df


def main():
    args = parse_args()
    
    # Load evaluation config
    with open(args.config, 'r') as f:
        eval_cfg = yaml.safe_load(f)
    
    # Load training config if specified
    training_cfg = None
    if "training_config" in eval_cfg:
        training_cfg = Config(eval_cfg["training_config"])
    
    # Setup directories
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(eval_cfg.get("output_dir", "output/evaluations")) / f"{eval_cfg['model']['type']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the evaluation config for reference
    with open(output_dir / "eval_config.yaml", 'w') as f:
        yaml.dump(eval_cfg, f)
    
    # Setup logger
    logger = setup_logger("eval", log_file=output_dir / "evaluation.log")
    logger.info(f"Evaluating {eval_cfg['model']['type']} model with checkpoint: {eval_cfg['checkpoint_path']}")
    
    # Set seed for reproducibility
    set_seed(eval_cfg.get("seed", 42))
    
    # Setup device
    device = torch.device(eval_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")
    
    # Determine data paths
    data_dir = Path(eval_cfg.get("data_dir", training_cfg.get("data.data_dir") if training_cfg else "data"))
    
    # Get dataset configs
    datasets = eval_cfg.get("datasets", ["val", "test"])
    batch_size = eval_cfg.get("batch_size", 8)
    
    # Load model
    model = load_model(
        eval_cfg["model"], 
        eval_cfg["checkpoint_path"],
        device,
        logger
    )
    
    results = {}
    
    # Process each dataset
    for dataset_name in datasets:
        if dataset_name not in eval_cfg["dataset_paths"]:
            logger.warning(f"Dataset '{dataset_name}' not found in config, skipping")
            continue
            
        csv_path = eval_cfg["dataset_paths"][dataset_name]
        logger.info(f"Evaluating on {dataset_name} dataset from {csv_path}")
        
        # Load dataset
        file_paths, ages, _, sexes, modalities = read_csv(csv_path, data_dir)
        if not file_paths:
            logger.error(f"No valid files found for {dataset_name} dataset")
            continue
            
        # Create data loader
        data_loader = create_data_loader(file_paths, ages, sexes, modalities, batch_size)
        
        # Evaluate model
        metrics, predictions, targets, sexes, modalities = evaluate_model(
            model, data_loader, device, eval_cfg["model"]["type"]
        )
        
        # Log results
        logger.info(f"{dataset_name.capitalize()} metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save predictions if requested
        if eval_cfg.get("save_predictions", False):
            output_path = output_dir / f"{dataset_name}_predictions.csv"
            save_predictions(file_paths, predictions, targets, sexes, modalities, output_path)
            logger.info(f"Saved {dataset_name} predictions to {output_path}")
        
        results[dataset_name] = {
            "metrics": metrics,
            "predictions": predictions,
            "targets": targets
        }
    
    # Save metrics summary
    summary = {
        "model_type": eval_cfg["model"]["type"],
        "checkpoint": eval_cfg["checkpoint_path"],
    }
    
    for dataset_name, result in results.items():
        summary[dataset_name] = result["metrics"]
    
    import json
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    
    # Return the main metrics for display
    result_summary = {}
    for dataset_name, result in results.items():
        result_summary[f"{dataset_name}_mae"] = result["metrics"]["mae"]
    
    return result_summary


if __name__ == "__main__":
    results = main()
    # Print final results to stdout for easy viewing
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")