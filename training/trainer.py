"""
Trainer class for brain age prediction models.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import logging
from pathlib import Path

from models.base_model import BrainAgeModel
from training.losses import get_loss_function
from training.metrics import calculate_metrics
from training.optimizers import get_optimizer, get_scheduler
from utils.wandb_logger import WandbLogger
from utils.logger import setup_logger


class BrainAgeTrainer:
    """
    Trainer for brain age prediction models with domain randomization.
    """
    
    def __init__(
        self,
        model: BrainAgeModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        use_wandb: bool = True,
        wandb_project: str = "brain-age-prediction",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            use_wandb: Whether to use Weights & Biases
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity name
            wandb_config: Weights & Biases configuration
            experiment_name: Name of the experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up experiment name
        self.experiment_name = experiment_name or f"{model.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.checkpoint_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = setup_logger(
            name=self.experiment_name,
            log_file=self.log_dir / f"{self.experiment_name}.log"
        )
        
        # Set up Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            self.wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=self.experiment_name,
                config=wandb_config or self.config
            )
        
        # Set up loss function
        self.criterion = get_loss_function(config.get("loss", "mse"))
        
        # Set up optimizer
        self.optimizer = get_optimizer(
            model.parameters(),
            optimizer_type=config.get("optimizer", "adam"),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5)
        )
        
        # Set up scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=config.get("scheduler", "cosine"),
            **config.get("scheduler_params", {})
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = config.get("epochs", 100)
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        # Mixed precision training
        self.use_amp = config.get("use_amp", True) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Log configuration
        self.logger.info(f"Initialized trainer for {self.experiment_name}")
        self.logger.info(f"Model: {model.model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"Batch size: {train_loader.batch_size}")
        self.logger.info(f"Using mixed precision: {self.use_amp}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            inputs = batch["image"].to(self.device)
            targets = batch["age"].float().to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
        
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        metrics = calculate_metrics(all_predictions, all_targets)
        metrics["loss"] = epoch_loss / len(self.train_loader)
        
        # Log metrics
        self.logger.info(f"Train Epoch: {epoch+1} | Loss: {metrics['loss']:.4f} | MAE: {metrics['mae']:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            self.wandb_logger.log({f"train/{k}": v for k, v in metrics.items()})
            self.wandb_logger.log({"lr": self.optimizer.param_groups[0]['lr']})
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Get data
                inputs = batch["image"].to(self.device)
                targets = batch["age"].float().to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                all_predictions.extend(outputs.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        metrics = calculate_metrics(all_predictions, all_targets)
        metrics["loss"] = val_loss / len(self.val_loader)
        
        # Log metrics
        self.logger.info(f"Val Epoch: {epoch+1} | Loss: {metrics['loss']:.4f} | MAE: {metrics['mae']:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            self.wandb_logger.log({f"val/{k}": v for k, v in metrics.items()})
            
            # Log sample predictions
            if epoch % 5 == 0 and len(all_predictions) > 0:
                self.wandb_logger.log_predictions(all_predictions[:20], all_targets[:20], epoch)
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary of training history
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "learning_rate": []
        }
        
        for epoch in range(self.epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_mae"].append(train_metrics["mae"])
            history["val_mae"].append(val_metrics["mae"])
            history["learning_rate"].append(self.optimizer.param_groups[0]['lr'])
            
            # Save checkpoint
            checkpoint_metrics = {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"]
            }
            
            extra_info = {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": epoch
            }
            
            _, is_best = self.model.save_checkpoint(
                checkpoint_dir=self.experiment_dir,
                metrics=checkpoint_metrics,
                extra_info=extra_info
            )
            
            # Early stopping
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.logger.info("Training completed")
        
        # Finalize wandb
        if self.use_wandb:
            self.wandb_logger.finish()
        
        return history