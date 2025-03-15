"""
Main training script for brain age prediction.
"""

import os
import torch
import numpy as np
import random
from pathlib import Path
import time
from datetime import datetime

from configs.config import Config, parse_args
from data.dataset import create_data_loaders
from dom_rand.domain_randomization import DomainRandomizer
from models.sfcn import SFCN
from models.resnet3d import ResNet3D
from training.trainer import BrainAgeTrainer
from utils.logger import setup_logger


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = Config(args.config)
    else:
        config = Config("configs/default.yaml")
    
    # Update configuration with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            if key.startswith("data_"):
                config.set(f"data.{key[5:]}", value)
            elif key.startswith("model_"):
                config.set(f"model.{key[6:]}", value)
            elif key.startswith("training_"):
                config.set(f"training.{key[9:]}", value)
            elif key.startswith("output_"):
                config.set(f"output.{key[7:]}", value)
            elif key.startswith("wandb_"):
                config.set(f"wandb.{key[6:]}", value)
            else:
                config.set(key, value)
    
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
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_csv=config.get("data.train_csv"),
        val_csv=config.get("data.val_csv"),
        image_dir=config.get("data.data_dir"),
        batch_size=config.get("training.batch_size", 8),
        num_workers=config.get("data.num_workers", 4),
        cache_dir=config.get("data.cache_dir"),
        domain_randomization_params=config.get("domain_randomization"),
        progressive_mode=config.get("domain_randomization.progressive_mode", False)
    )
    
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
        #model = EfficientNet3D(**model_params)
        pass
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