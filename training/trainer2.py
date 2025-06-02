import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, StepLR
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from brain_age_pred.utils.logger import setup_logger
import wandb
from brain_age_pred.utils.wandb_logger import WandbLogger
import logging.handlers

class BrainAgeTrainer:
    """
    Comprehensive trainer for brain age prediction with detailed metrics and wandb integration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
        use_wandb: bool = True,
        wandb_project: str = "brain-age-pred",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        experiment_name: str = "experiment",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_config = wandb_config
        self.experiment_name = experiment_name
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
        self.history = {"train": [], "val": []}
        
        # Setup logger with wandb integration
        self.logger = setup_logger(
            name=self.experiment_name,
            log_file=self.log_dir / f"{self.experiment_name}.log"
        )
        
        # Add wandb handler to logger if wandb is enabled
        if use_wandb:
            self.wandb = WandbLogger(
                project = wandb_project,
                entity  = wandb_entity,
                name    = self.experiment_name,
                config  = wandb_config or {},
            )
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_amp()
        
        # Age brackets for detailed analysis (10-year gaps)
        self.age_brackets = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
        
        self.logger.info(f"BrainAgeTrainer initialized for {experiment_name}")
        
    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = self.config.get("momentum", 0.9)
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        self.logger.info(f"Optimizer: {optimizer_name}, LR: {lr}, Weight Decay: {weight_decay}")
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler based on config."""
        scheduler_name = self.config.get("scheduler", "none").lower()
        scheduler_params = self.config.get("scheduler_params", {})
        
        if scheduler_name == "cosine":
            T_max = scheduler_params.get("T_max", self.config.get("epochs", 100))
            eta_min = scheduler_params.get("eta_min", 1e-6)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_name == "plateau":
            patience = scheduler_params.get("patience", 10)
            factor = scheduler_params.get("factor", 0.5)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=patience, factor=factor)
        elif scheduler_name == "onecycle":
            max_lr = scheduler_params.get("max_lr", self.config.get("learning_rate", 1e-3))
            epochs = scheduler_params.get("epochs", self.config.get("epochs", 100))
            steps_per_epoch = scheduler_params.get("steps_per_epoch", len(self.train_loader))
            self.scheduler = OneCycleLR(self.optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
        elif scheduler_name == "step":
            step_size = scheduler_params.get("step_size", 30)
            gamma = scheduler_params.get("gamma", 0.1)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            self.scheduler = None
            
        self.logger.info(f"Scheduler: {scheduler_name}")
        
    def _setup_loss_function(self):
        """Setup loss function based on config."""
        loss_name = self.config.get("loss", "mae").lower()
        loss_params = self.config.get("loss_params", {})
        
        if loss_name == "mae":
            self.criterion = nn.L1Loss()
        elif loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "huber":
            delta = loss_params.get("delta", 1.0)
            self.criterion = nn.HuberLoss(delta=delta)
        elif loss_name == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
            
        self.logger.info(f"Loss function: {loss_name}")
        
    def _setup_amp(self):
        """Setup Automatic Mixed Precision if enabled."""
        self.use_amp = self.config.get("use_amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("AMP enabled")
        else:
            self.scaler = None
            
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                        modalities: Optional[List[str]] = None, 
                        sexes: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute comprehensive metrics for brain age prediction."""
        metrics = {}
        
        # Overall metrics
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        correlation, _ = pearsonr(targets, predictions) if len(targets) > 1 else (0.0, 1.0)
        
        metrics.update({
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "correlation": correlation,
        })
        
        # Age bracket analysis
        for min_age, max_age in self.age_brackets:
            mask = (targets >= min_age) & (targets < max_age)
            if np.sum(mask) > 0:
                bracket_mae = mean_absolute_error(targets[mask], predictions[mask])
                metrics[f"mae_age_{min_age}_{max_age}"] = bracket_mae
            else:
                metrics[f"mae_age_{min_age}_{max_age}"] = 0.0
                
        # Modality analysis
        if modalities is not None:
            unique_modalities = list(set(modalities))
            for modality in unique_modalities:
                mask = np.array([m == modality for m in modalities])
                if np.sum(mask) > 0:
                    modality_mae = mean_absolute_error(targets[mask], predictions[mask])
                    metrics[f"mae_modality_{modality}"] = modality_mae
                    
        # Sex analysis
        if sexes is not None:
            unique_sexes = list(set(sexes))
            for sex in unique_sexes:
                mask = np.array([s == sex for s in sexes])
                if np.sum(mask) > 0:
                    sex_mae = mean_absolute_error(targets[mask], predictions[mask])
                    metrics[f"mae_sex_{sex}"] = sex_mae
                    
        return metrics
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        modalities = []
        sexes = []
        
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            ages = batch["age"].to(self.device, non_blocking=True)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(), ages)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), ages)
                
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
            total_loss += loss.item() * gradient_accumulation_steps * batch_size
            
            # Collect predictions for metrics
            with torch.no_grad():
                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(ages.cpu().numpy())
                if "modality" in batch:
                    modalities.extend(batch["modality"])
                if "sex" in batch:
                    sexes.extend(batch["sex"])
                    
        # Step scheduler if not plateau-based
        if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
            if isinstance(self.scheduler, OneCycleLR):
                # OneCycleLR steps every batch
                pass  # Already stepped in training loop if implemented
            else:
                self.scheduler.step()
                
        avg_loss = total_loss / total_samples
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = self._compute_metrics(
            predictions, targets, 
            modalities if modalities else None,
            sexes if sexes else None
        )
        metrics["loss"] = avg_loss
        
        return metrics
        
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        modalities = []
        sexes = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                ages = batch["age"].to(self.device, non_blocking=True)
                
                batch_size = images.size(0)
                total_samples += batch_size
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs.squeeze(), ages)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(), ages)
                    
                total_loss += loss.item() * batch_size
                
                # Collect predictions for metrics
                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(ages.cpu().numpy())
                if "modality" in batch:
                    modalities.extend(batch["modality"])
                if "sex" in batch:
                    sexes.extend(batch["sex"])
                    
        avg_loss = total_loss / total_samples
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = self._compute_metrics(
            predictions, targets,
            modalities if modalities else None,
            sexes if sexes else None
        )
        metrics["loss"] = avg_loss
        
        return metrics
        
    def _save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "metrics": metrics,
            "config": self.config,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation MAE: {metrics['mae']:.4f}")
            
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to wandb and console."""
        # Console logging
        self.logger.info(
            f"Epoch {self.current_epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train MAE: {train_metrics['mae']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R²: {val_metrics['r2']:.4f} | "
            f"Val Corr: {val_metrics['correlation']:.4f}"
        )
        
        # WandB logging
        if self.use_wandb:
            wandb_log = {}
            
            # Add train metrics with prefix
            for key, value in train_metrics.items():
                wandb_log[f"train/{key}"] = value
                
            # Add validation metrics with prefix
            for key, value in val_metrics.items():
                wandb_log[f"val/{key}"] = value
                
            # Add learning rate
            if self.optimizer.param_groups:
                wandb_log["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                
            wandb_log["epoch"] = self.current_epoch
            wandb.log(wandb_log, step=self.current_epoch)
            
    def train(self) -> Dict[str, List]:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        epochs = self.config.get("epochs", 100)
        early_stopping_patience = self.config.get("early_stopping_patience", 20)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training
            start_time = time.time()
            train_metrics = self._train_epoch()
            train_time = time.time() - start_time
            
            # Validation
            start_time = time.time()
            val_metrics = self._validate_epoch()
            val_time = time.time() - start_time
            
            # Update history
            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)
            
            # Step plateau scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["mae"])
                
            # Check for best model
            current_metric = val_metrics["mae"]  # Use MAE as primary metric
            is_best = current_metric < self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
            # Save checkpoint
            self._save_checkpoint(val_metrics, is_best)
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Early stopping
            if self.early_stopping_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
                
            # Log timing
            self.logger.info(f"Epoch {epoch} timings - Train: {train_time:.2f}s, Val: {val_time:.2f}s")
            
        self.logger.info(f"Training completed. Best validation MAE: {self.best_metric:.4f} at epoch {self.best_epoch}")
        return self.history
        
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.logger.info("Starting evaluation...")
        
        # Load best model
        best_checkpoint_path = self.checkpoint_dir / "best.pth"
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info("Loaded best model for evaluation")
        else:
            self.logger.warning("Best checkpoint not found, using current model state")
            
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        modalities = []
        sexes = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                ages = batch["age"].to(self.device, non_blocking=True)
                
                batch_size = images.size(0)
                total_samples += batch_size
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs.squeeze(), ages)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(), ages)
                    
                total_loss += loss.item() * batch_size
                
                # Collect predictions for metrics
                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(ages.cpu().numpy())
                if "modality" in batch:
                    modalities.extend(batch["modality"])
                if "sex" in batch:
                    sexes.extend(batch["sex"])
                    
        avg_loss = total_loss / total_samples
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = self._compute_metrics(
            predictions, targets,
            modalities if modalities else None,
            sexes if sexes else None
        )
        metrics["loss"] = avg_loss
        
        # Log evaluation results
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Loss: {metrics['loss']:.4f}")
        self.logger.info(f"  MAE: {metrics['mae']:.4f}")
        self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        self.logger.info(f"  R²: {metrics['r2']:.4f}")
        self.logger.info(f"  Correlation: {metrics['correlation']:.4f}")
        
        # Log age bracket results
        self.logger.info("Age Bracket MAE:")
        for min_age, max_age in self.age_brackets:
            bracket_key = f"mae_age_{min_age}_{max_age}"
            if bracket_key in metrics:
                self.logger.info(f"  Ages {min_age}-{max_age}: {metrics[bracket_key]:.4f}")
                
        return metrics



# ======================================
# Instrumented version of BrainAgeTrainer with extensive
# gradient-flow debugging utilities. Save this file as
# `brain_age_trainer_debug.py` next to your project and
# import `BrainAgeTrainerDebug` instead of `BrainAgeTrainer`.
# --------------------------------------
#  Usage example
#  -------------
#  from brain_age_trainer_debug import BrainAgeTrainerDebug
#  trainer = BrainAgeTrainerDebug(model, train_loader, val_loader,
#                                 test_loader, config, device,
#                                 checkpoint_dir, log_dir,
#                                 use_wandb=True)
#  history = trainer.train()
# ======================================
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import wandb

class BrainAgeTrainerDebug(BrainAgeTrainer):
    """Extension of BrainAgeTrainer that prints / logs detailed gradient-flow
    statistics and activation distributions each optimisation step.
    Designed for quick sanity-checks (e.g., over-fitting 10 MRI volumes).
    """

    def __init__(self, *args, gradient_print_freq: int = 1, **kwargs):
        """Extra argument
        Parameters
        ----------
        gradient_print_freq : int
            Frequency (in optimiser steps) at which gradient statistics are
            reported. Set to 1 for every step; increase to reduce verbosity.
        """
        super().__init__(*args, **kwargs)
        self._grad_step_idx = 0
        self.gradient_print_freq = gradient_print_freq

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect_grad_stats(self):
        """Return dictionaries with gradient norms and zero-grad counts."""
        grad_norms: Dict[str, float] = {}
        zero_grads = 0
        total_params = 0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total_params += g.numel()
            zero_grads += (g == 0).sum().item()
            grad_norms[name] = g.norm().item()
        return grad_norms, zero_grads, total_params

    @torch.no_grad()
    def _log_gradient_flow(self):
        """Print / log gradient norms & zero-gradient percentage."""
        grad_norms, zero_grads, total = self._collect_grad_stats()
        max_norm = max(grad_norms.values()) if grad_norms else 0.0
        mean_norm = float(np.mean(list(grad_norms.values()))) if grad_norms else 0.0
        pct_zero = 100.0 * zero_grads / total if total else 0.0

        msg = (f"[Grad-Flow] step={self._grad_step_idx:04d} | max_norm={max_norm:.3e} "
               f"| mean_norm={mean_norm:.3e} | zero_grad={pct_zero:.2f}%")
        self.logger.info(msg)

        # Optional: log histogram to WandB
        if self.use_wandb and (self._grad_step_idx % self.gradient_print_freq == 0):
            wandb_data = {
                "grad_flow/max_norm": max_norm,
                "grad_flow/mean_norm": mean_norm,
                "grad_flow/zero_grad_pct": pct_zero,
                "grad_flow/step": self._grad_step_idx,
            }
            # Also send per-layer norms (only small models, else comment out)
            for k, v in grad_norms.items():
                wandb_data[f"grad_norm/{k}"] = v
            wandb.log(wandb_data, step=self._grad_step_idx)

    # ------------------------------------------------------------------
    #  Override training step to insert debug hooks
    # ------------------------------------------------------------------
    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        predictions, targets, modalities, sexes = [], [], [], []

        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        for batch_idx, batch in enumerate(self.train_loader):
            imgs = batch["image"].to(self.device, non_blocking=True)
            ages = batch["age"].to(self.device, non_blocking=True)
            bsz = imgs.size(0)
            total_samples += bsz

            # Forward
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outs = self.model(imgs)
                    loss = self.criterion(outs.squeeze(), ages)
            else:
                outs = self.model(imgs)
                loss = self.criterion(outs.squeeze(), ages)

            loss_scaled = loss / gradient_accumulation_steps

            # Backward
            if self.use_amp:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Optimiser step & zero-grad every accumulation interval
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                # ------------------------------------------------------
                #  DEBUG: gradient-flow statistics
                # ------------------------------------------------------
                if (self._grad_step_idx % self.gradient_print_freq) == 0:
                    self._log_gradient_flow()
                self._grad_step_idx += 1

            total_loss += loss.item() * bsz

            # Collect preds for epoch-level metrics
            with torch.no_grad():
                predictions.extend(outs.squeeze().cpu().numpy())
                targets.extend(ages.cpu().numpy())
                if "modality" in batch:
                    modalities.extend(batch["modality"])
                if "sex" in batch:
                    sexes.extend(batch["sex"])

        # Scheduler (if not ReduceLROnPlateau)
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

        avg_loss = total_loss / total_samples
        metrics = self._compute_metrics(np.array(predictions), np.array(targets),
                                        modalities if modalities else None,
                                        sexes if sexes else None)
        metrics["loss"] = avg_loss

        return metrics
