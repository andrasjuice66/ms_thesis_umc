"""
Main training script for brain age prediction.
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
import time
from datetime import datetime
import pandas as pd

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now import from project modules
from brain_age_pred.configs.config import Config
from brain_age_pred.dom_rand.dataset import BrainMRIDataset
from brain_age_pred.dom_rand.domain_randomization import BrainAgeDomainRandomizer
from brain_age_pred.models.sfcn import SFCN
from brain_age_pred.models.resnet3d import ResNet3D
from brain_age_pred.models.efficientnet3d import EfficientNet3D
from brain_age_pred.training.trainer import BrainAgeTrainer
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed  


def load_data_from_csv(csv_path, image_dir):
    """Load file paths and age labels from CSV file."""
    df = pd.read_csv(csv_path)
    file_paths = []
    age_labels = []
    
    for _, row in df.iterrows():
        file_name = row['filename']  # Adjust column name if needed
        age = row['age']  # Adjust column name if needed
        file_path = os.path.join(image_dir, file_name)
        
        if os.path.exists(file_path):
            file_paths.append(file_path)
            age_labels.append(age)
    
    return file_paths, age_labels


def main():
    """Main function."""
    # Load configuration directly from default.yaml
    config = Config("configs/default.yaml")
    
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Set device
    device = config.get("device")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Create experiment name and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("output.experiment_name")
    if experiment_name is None:
        model_type = config.get("model.type", "sfcn")
        augmentation_strength = config.get("domain_randomization.augmentation_strength", "medium")
        experiment_name = f"{model_type}_{augmentation_strength}_{timestamp}"
    
    output_dir = Path(config.get("output.output_dir", "output"))
    checkpoint_dir = output_dir / config.get("output.checkpoint_dir", "checkpoints") / experiment_name
    log_dir = output_dir / config.get("output.log_dir", "logs") / experiment_name
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(
        name="brain-age-pred",
        log_file=log_dir / "train.log"
    )
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config}")
    
    # Save configuration
    config.save_config(checkpoint_dir / "config.yaml")
    
    # ======== NEW DATA LOADING APPROACH WITH OPTIMIZED CLASSES ========
    logger.info("Creating data loaders with optimized dataset...")
    
    # Get domain randomization parameters from config
    dr_params = config.get("domain_randomization", {})
    
    # Create domain randomizer with the configured parameters
    randomizer = BrainAgeDomainRandomizer(
        image_key="image",
        age_key="age",
        # Intensity parameters
        intensity_bounds=dr_params.get("intensity_bounds", (0, 100)),
        contrast_bounds=dr_params.get("contrast_bounds", (0.75, 1.25)),
        # Spatial parameters
        flipping=dr_params.get("flipping", True),
        scaling_bounds=dr_params.get("scaling_bounds", 0.2),
        rotation_bounds=dr_params.get("rotation_bounds", 15),
        shearing_bounds=dr_params.get("shearing_bounds", 0.012),
        nonlin_std=dr_params.get("nonlin_std", 3.0),
        # Resampling parameters
        randomise_res=dr_params.get("randomise_res", True),
        max_res_iso=dr_params.get("max_res_iso", 3.0),
        max_res_aniso=dr_params.get("max_res_aniso", 5.0),
        # Bias field parameters
        bias_field_std=dr_params.get("bias_field_std", 0.5),
        # Progressive augmentation parameters
        progressive_mode=dr_params.get("progressive_mode", False),
        current_epoch=0,
        max_epochs=config.get("training.epochs", 100),
        # Performance parameters
        device=device,
        num_workers=config.get("data.num_workers", 4),
        prefetch_factor=config.get("data.prefetch_factor", 2),
        use_cache=dr_params.get("use_cache", True),
        cache_size=dr_params.get("cache_size", 100)
    )
    
    # Load paths and labels from CSV
    train_files, train_ages = load_data_from_csv(
        config.get("data.train_csv"),
        config.get("data.data_dir")
    )
    
    val_files, val_ages = load_data_from_csv(
        config.get("data.val_csv"),
        config.get("data.data_dir")
    )
    
    # Create train dataset with domain randomization enabled
    train_dataset = BrainMRIDataset(
        file_paths=train_files,
        age_labels=train_ages,
        transform=randomizer,
        cache_mode=config.get("data.cache_mode", "memory"),
        cache_dir=config.get("data.cache_dir"),
        target_shape=config.get("data.target_shape", (176, 240, 256)),
        preload=config.get("data.preload", True),
        normalize=config.get("data.normalize", True),
        device=torch.device("cpu"),  # Keep raw data on CPU, transformations handle device
        mode="train"  # Explicitly set to training mode
    )
    
    # Create validation dataset with the same randomizer but in "val" mode
    # The mode="val" will ensure no randomization is applied regardless of the randomizer
    val_dataset = BrainMRIDataset(
        file_paths=val_files,
        age_labels=val_ages,
        transform=randomizer,  # Can use the same randomizer - it won't be applied in val mode
        cache_mode=config.get("data.cache_mode", "memory"),
        cache_dir=config.get("data.cache_dir", None),
        target_shape=config.get("data.target_shape", (176, 240, 256)),
        preload=config.get("data.preload", True),
        normalize=config.get("data.normalize", True),
        device=torch.device("cpu"),
        mode="val"  # Explicitly set to validation mode - no randomization
    )
    
    # Create dataloaders with optimized settings
    train_loader = train_dataset.create_dataloader(
        batch_size=config.get("training.batch_size", 8),
        shuffle=True,
        num_workers=config.get("data.num_workers", 4),
        pin_memory=device.type == "cuda",
        persistent_workers=config.get("data.persistent_workers", True) and config.get("data.num_workers", 4) > 0,
        prefetch_factor=config.get("data.prefetch_factor", 2)
    )
    
    val_loader = val_dataset.create_dataloader(
        batch_size=config.get("training.batch_size", 8),
        shuffle=False,
        num_workers=config.get("data.num_workers", 4),
        pin_memory=device.type == "cuda",
        persistent_workers=config.get("data.persistent_workers", True) and config.get("data.num_workers", 4) > 0,
        prefetch_factor=config.get("data.prefetch_factor", 2)
    )
    
    logger.info(f"Created training dataset with {len(train_dataset)} samples")
    logger.info(f"Created validation dataset with {len(val_dataset)} samples")
    
    # Create model
    logger.info("Creating model...")
    model_type = config.get("model.type", "sfcn")
    model_params = {
        "in_channels": config.get("model.in_channels", 1),
        "dropout_rate": config.get("model.dropout_rate", 0.3),
        "use_attention": config.get("model.use_attention", False)
    }
    
    if model_type == "sfcn":
        model = SFCN(**model_params)
    elif model_type == "resnet3d":
        model = ResNet3D(**model_params)
    elif model_type == "efficientnet3d":
        model = EfficientNet3D(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint if provided
    checkpoint_path = config.get("model.checkpoint")
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        model = model.__class__.load_from_checkpoint(checkpoint_path, map_location=device)
    
    model = model.to(device)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = BrainAgeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.get("training"),
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        use_wandb=config.get("wandb.use_wandb", True),
        wandb_project=config.get("wandb.project", "brain-age-prediction"),
        wandb_entity=config.get("wandb.entity"),
        wandb_config=config.config,
        experiment_name=experiment_name
    )
    
    # Train model
    logger.info("Starting training...")
    start_time = time.time()
    history = trainer.train()
    end_time = time.time()
    
    # Log training time
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save training history
    import json
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)
    
    logger.info(f"Training history saved to {checkpoint_dir / 'history.json'}")
    logger.info("Done!")


if __name__ == "__main__":
    main() 