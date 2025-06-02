
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
    Comprehensive trainer for brain age prediction with detailed metrics, wandb integration, and gradient debugging.
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
        debug_gradients: bool = True,
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
        self.debug_gradients = debug_gradients
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
        self.history = {"train": [], "val": []}
        
        # Debug state
        self._debug_step = 0
        self.activations = {}
        self.gradients = {}
        self.prev_params = {}
        
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
        
        # Setup debugging hooks if enabled
        if self.debug_gradients:
            self._setup_debug_hooks()
        
        # Age brackets for detailed analysis (10-year gaps)
        self.age_brackets = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
        
        self.logger.info(f"BrainAgeTrainer initialized for {experiment_name}")
        if self.debug_gradients:
            self.logger.info("Gradient debugging enabled")
        
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
    
    def _setup_debug_hooks(self):
        """Setup hooks for debugging activations and gradients."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        def get_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach()
            return hook

        # Register hooks for Conv3d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                module.register_forward_hook(get_activation(name))
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.register_hook(get_gradient(f"{name}_weight"))
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.register_hook(get_gradient(f"{name}_bias"))
    
    def log_gradient_norms(self):
        """Log gradient norms for each layer."""
        total_norm = 0
        param_count = 0
        gradient_dict = {}
        dead_params = 0
        total_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                gradient_dict[f"grad_norm/{name}"] = param_norm
                
                # Check for dead gradients
                if param_norm < 1e-8:
                    dead_params += param.numel()
                total_params += param.numel()
            else:
                gradient_dict[f"grad_norm/{name}"] = 0.0
                dead_params += param.numel() if param.numel() else 0
                total_params += param.numel() if param.numel() else 0

        total_norm = total_norm ** 0.5 if param_count > 0 else 0
        gradient_dict["grad_norm/total"] = total_norm
        gradient_dict["grad_norm/dead_ratio"] = dead_params / total_params if total_params > 0 else 0

        return gradient_dict
    
    def track_parameter_changes(self):
        """Track parameter changes between updates."""
        changes = {}
        total_change = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if name in self.prev_params:
                change = torch.norm(param.data - self.prev_params[name]).item()
                changes[f"param_change/{name}"] = change
                total_change += change
                param_count += 1
        
        if param_count > 0:
            changes["param_change/total"] = total_change
            changes["param_change/average"] = total_change / param_count
        
        return changes
    
    def log_activations_and_gradients(self):
        """Log statistics of activations and gradients."""
        stats = {}

        # Activation statistics
        for name, activation in self.activations.items():
            if activation is not None:
                stats[f"activation_mean/{name}"] = activation.mean().item()
                stats[f"activation_std/{name}"] = activation.std().item()
                stats[f"activation_max/{name}"] = activation.max().item()
                stats[f"activation_min/{name}"] = activation.min().item()
                
                # Check for dead activations (all zeros)
                dead_ratio = (activation == 0).float().mean().item()
                stats[f"activation_dead_ratio/{name}"] = dead_ratio

        # Gradient statistics from hooks
        for name, gradient in self.gradients.items():
            if gradient is not None:
                stats[f"hook_gradient_mean/{name}"] = gradient.mean().item()
                stats[f"hook_gradient_std/{name}"] = gradient.std().item()
                stats[f"hook_gradient_max/{name}"] = gradient.max().item()
                stats[f"hook_gradient_min/{name}"] = gradient.min().item()

        return stats
    
    def debug_training_step(self, batch_idx, loss):
        """Comprehensive debugging for training step."""
        debug_freq = self.config.get("debug_frequency", 10)
        
        if batch_idx % debug_freq == 0:
            debug_stats = {}
            
            # 1. Loss debugging
            debug_stats["debug/loss"] = loss.item()
            debug_stats["debug/step"] = self._debug_step
            
            # 2. Gradient norms
            grad_stats = self.log_gradient_norms()
            debug_stats.update(grad_stats)
            
            # 3. Parameter changes (if we have previous params)
            if self.prev_params:
                param_changes = self.track_parameter_changes()
                debug_stats.update(param_changes)
            
            # 4. Activation and gradient statistics
            if self.debug_gradients:
                act_grad_stats = self.log_activations_and_gradients()
                debug_stats.update(act_grad_stats)
            
            # 5. Learning rate
            if self.optimizer.param_groups:
                debug_stats["debug/learning_rate"] = self.optimizer.param_groups[0]["lr"]
            
            # Console logging for immediate feedback
            print(f"\\n=== DEBUG STEP {self._debug_step} (Batch {batch_idx}) ===")
            print(f"Loss: {loss.item():.6f}")
            print(f"Total Grad Norm: {debug_stats.get('grad_norm/total', 0):.2e}")
            print(f"Dead Gradient Ratio: {debug_stats.get('grad_norm/dead_ratio', 0):.4f}")
            if 'param_change/total' in debug_stats:
                print(f"Total Param Change: {debug_stats['param_change/total']:.2e}")
            print("=" * 50)
            
            # WandB logging
            if self.use_wandb:
                wandb.log(debug_stats, step=self._debug_step)
        
        self._debug_step += 1
    
    def save_parameter_snapshot(self):
        """Save current parameters for change tracking."""
        self.prev_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
            
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
        """Train for one epoch with comprehensive debugging."""
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
            
            # === CRITICAL DEBUG: Save parameter snapshot BEFORE any computation ===
            if batch_idx == 0:  # First batch of epoch
                print(f"\\n=== EPOCH {self.current_epoch} BATCH {batch_idx} PARAMETER DEBUG ===")
                # Save a few key parameters for comparison
                self.debug_params_before = {}
                param_count = 0
                for name, param in self.model.named_parameters():
                    if param_count < 3:  # Just first 3 parameters
                        self.debug_params_before[name] = param.clone().detach()
                        print(f"BEFORE - {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                    param_count += 1
                    if param_count >= 3:
                        break
            
            # Save parameter snapshot for change tracking (existing debug)
            if self.debug_gradients and batch_idx % self.config.get("debug_frequency", 10) == 0:
                self.save_parameter_snapshot()
            
            # === FORWARD PASS ===
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(), ages)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), ages)
                
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # === CRITICAL DEBUG: Check loss and gradients ===
            if batch_idx == 0:
                print(f"Loss before backward: {loss.item():.6f}")
                print(f"Loss requires_grad: {loss.requires_grad}")
                print(f"Model training mode: {self.model.training}")
            
            # === BACKWARD PASS ===
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # === CRITICAL DEBUG: Check gradients before optimizer step ===
                    if batch_idx == 0:
                        print("\\n=== GRADIENT CHECK BEFORE OPTIMIZER STEP ===")
                        grad_count = 0
                        for name, param in self.model.named_parameters():
                            if grad_count < 3:
                                if param.grad is not None:
                                    grad_norm = param.grad.norm().item()
                                    print(f"GRAD - {name}: norm={grad_norm:.2e}, mean={param.grad.mean().item():.2e}")
                                else:
                                    print(f"GRAD - {name}: NO GRADIENT!")
                            grad_count += 1
                            if grad_count >= 3:
                                break
                    
                    # Debug before optimizer step
                    if self.debug_gradients:
                        self.debug_training_step(batch_idx, loss)
                    
                    # === CRITICAL DEBUG: Check optimizer state ===
                    if batch_idx == 0:
                        print(f"\\nOptimizer LR: {self.optimizer.param_groups[0]['lr']}")
                        print(f"Optimizer state_dict keys: {list(self.optimizer.state_dict().keys())}")
                    
                    # === OPTIMIZER STEP ===
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # === CRITICAL DEBUG: Check parameters AFTER optimizer step ===
                    if batch_idx == 0:
                        print("\\n=== PARAMETER CHECK AFTER OPTIMIZER STEP ===")
                        total_change = 0.0
                        param_count = 0
                        for name, param in self.model.named_parameters():
                            if param_count < 3 and name in self.debug_params_before:
                                old_param = self.debug_params_before[name]
                                change = torch.norm(param - old_param).item()
                                total_change += change
                                print(f"AFTER - {name}: change={change:.2e}")
                                print(f"AFTER - {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                            param_count += 1
                            if param_count >= 3:
                                break
                        print(f"TOTAL PARAMETER CHANGE: {total_change:.2e}")
                        
                        # Additional check: verify optimizer actually ran
                        print(f"\\nOptimizer step count check:")
                        if hasattr(self.optimizer, 'state'):
                            print(f"Optimizer has state: {len(self.optimizer.state) > 0}")
                            if len(self.optimizer.state) > 0:
                                first_param_id = next(iter(self.optimizer.state.keys()))
                                state = self.optimizer.state[first_param_id]
                                print(f"First param state keys: {list(state.keys())}")
                        
            else:
                loss.backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # === CRITICAL DEBUG: Check gradients before optimizer step ===
                    if batch_idx == 0:
                        print("\\n=== GRADIENT CHECK BEFORE OPTIMIZER STEP ===")
                        grad_count = 0
                        total_grad_norm = 0.0
                        for name, param in self.model.named_parameters():
                            if grad_count < 3:
                                if param.grad is not None:
                                    grad_norm = param.grad.norm().item()
                                    total_grad_norm += grad_norm
                                    print(f"GRAD - {name}: norm={grad_norm:.2e}, mean={param.grad.mean().item():.2e}")
                                    print(f"GRAD - {name}: max={param.grad.max().item():.2e}, min={param.grad.min().item():.2e}")
                                else:
                                    print(f"GRAD - {name}: NO GRADIENT!")
                            grad_count += 1
                            if grad_count >= 3:
                                break
                        print(f"TOTAL GRAD NORM (first 3 layers): {total_grad_norm:.2e}")
                    
                    # Debug before optimizer step
                    if self.debug_gradients:
                        self.debug_training_step(batch_idx, loss)
                    
                    # === CRITICAL DEBUG: Check optimizer state ===
                    if batch_idx == 0:
                        print(f"\\nOptimizer type: {type(self.optimizer).__name__}")
                        print(f"Optimizer LR: {self.optimizer.param_groups[0]['lr']}")
                        print(f"Optimizer weight_decay: {self.optimizer.param_groups[0].get('weight_decay', 'N/A')}")
                        print(f"Number of param groups: {len(self.optimizer.param_groups)}")
                        print(f"Number of parameters in first group: {len(self.optimizer.param_groups[0]['params'])}")
                    
                    # === SAVE PARAMS BEFORE OPTIMIZER STEP ===
                    if batch_idx == 0:
                        params_before_step = {}
                        for name, param in self.model.named_parameters():
                            params_before_step[name] = param.clone().detach()
                    
                    # === OPTIMIZER STEP ===
                    print(f"\\n=== CALLING OPTIMIZER.STEP() ===")
                    self.optimizer.step()
                    print(f"=== OPTIMIZER.STEP() COMPLETED ===")
                    
                    # === CRITICAL DEBUG: Check parameters AFTER optimizer step ===
                    if batch_idx == 0:
                        print("\\n=== PARAMETER CHECK AFTER OPTIMIZER STEP ===")
                        total_change = 0.0
                        max_change = 0.0
                        param_count = 0
                        
                        for name, param in self.model.named_parameters():
                            if name in params_before_step:
                                old_param = params_before_step[name]
                                change = torch.norm(param - old_param).item()
                                total_change += change
                                max_change = max(max_change, change)
                                
                                if param_count < 3:
                                    print(f"CHANGE - {name}: {change:.2e}")
                                    print(f"AFTER  - {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                                    
                                    # Check if any values actually changed
                                    diff = param - old_param
                                    nonzero_changes = (diff.abs() > 1e-10).sum().item()
                                    total_elements = param.numel()
                                    print(f"ELEMENTS CHANGED - {name}: {nonzero_changes}/{total_elements} ({100*nonzero_changes/total_elements:.2f}%)")
                                
                                param_count += 1
                        
                        print(f"\\nSUMMARY:")
                        print(f"TOTAL PARAMETER CHANGE: {total_change:.2e}")
                        print(f"MAX SINGLE PARAM CHANGE: {max_change:.2e}")
                        
                        # Check if gradients were consumed
                        print(f"\\nGRADIENT CHECK AFTER OPTIMIZER STEP:")
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                print(f"WARNING: {name} still has gradients after step!")
                            break  # Just check first param
                    
                    # === ZERO GRADIENTS ===
                    self.optimizer.zero_grad()
                    
                    # === FINAL VERIFICATION ===
                    if batch_idx == 0:
                        print(f"\\n=== FINAL VERIFICATION ===")
                        print(f"Gradients zeroed: {all(p.grad is None or p.grad.norm().item() < 1e-10 for p in self.model.parameters() if p.grad is not None)}")
                        print("=" * 80)
                        
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

