from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_age_pred.training.losses import get_loss_function
from brain_age_pred.training.optimizers import get_optimizer, get_scheduler
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.wandb_logger import WandbLogger

class MultiTaskTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
        use_wandb: bool = False,
        wandb_project: str = "brain-age-multitask",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.exp_name = experiment_name or f"{model.__class__.__name__}_{int(time.time())}"
        self.logger = setup_logger(self.exp_name, self.log_dir / f"{self.exp_name}.log")

        self.use_wandb = use_wandb
        if use_wandb:
            self.wandb = WandbLogger(project=wandb_project, entity=wandb_entity, name=self.exp_name, config=wandb_config or {})

        # Losses
        loss_cfg = self.cfg.get("loss", {})
        self.age_loss_name = loss_cfg.get("age_loss_fn", "mae")
        self.age_criterion = get_loss_function(self.age_loss_name)
        self.seg_criterion = DiceCELoss(to_onehot_y=True, softmax=True)

        # Uncertainty-based loss balancing
        self.log_var_age = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_var_seg = nn.Parameter(torch.zeros(1, device=self.device))
        
        # Optimizer and Scheduler
        params_to_optimize = list(model.parameters()) + [self.log_var_age, self.log_var_seg]
        self.optimizer = get_optimizer(params_to_optimize, **self.cfg.get("optimizer", {}))
        self.scheduler = get_scheduler(self.optimizer, **self.cfg.get("scheduler", {}))

        # Training settings
        self.epochs = self.cfg.get("epochs", 100)
        self.grad_accum_steps = self.cfg.get("gradient_accumulation_steps", 1)
        self.early_stopping_patience = self.cfg.get("early_stopping_patience", 20)
        self.use_amp = self.cfg.get("use_amp", True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.best_val_mae = float("inf")
        self.best_mae_epoch = -1
        self.best_mae_checkpoint_path = None
        self.early_stop_counter = 0

        self.model.to(self.device)
        self.logger.info(f"Trainer initialized for experiment: {self.exp_name}")

    def switch_to_age_only_mode(self, freeze_encoder=False):
        """Switch from multitask training to age-only fine-tuning.
        
        Args:
            freeze_encoder: If True, freeze encoder too (only age head trainable)
                           If False, keep encoder trainable (encoder + age head trainable)
        """
        self.logger.info("Switching to age-only training mode...")
        
        # Always freeze segmentation head parameters
        for param in self.model.seg_head.parameters():
            param.requires_grad = False
        
        params_to_optimize = []
        
        if freeze_encoder:
            # Freeze encoder too - only age head trainable
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            self.logger.info("Encoder frozen - only age head will be fine-tuned")
        else:
            # Keep encoder trainable
            params_to_optimize.extend(self.model.encoder.parameters())
            self.logger.info("Encoder trainable - encoder + age head will be fine-tuned")
        
        # Always add age head parameters (this is what we want to fine-tune)
        params_to_optimize.extend(self.model.age_head.parameters())
        
        # Only keep age uncertainty parameter (remove seg uncertainty)
        params_to_optimize.append(self.log_var_age)
        
        # Recreate optimizer and scheduler with only trainable parameters
        self.optimizer = get_optimizer(params_to_optimize, **self.cfg.get("optimizer", {}))
        self.scheduler = get_scheduler(self.optimizer, **self.cfg.get("scheduler", {}))
        
        # Set flag for age-only mode
        self.age_only_mode = True
        
        self.logger.info(f"Age-only mode activated. Optimizing {len(params_to_optimize)} parameters")
        
        # Log detailed parameter breakdown
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Break down by component
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        seg_head_params = sum(p.numel() for p in self.model.seg_head.parameters())
        age_head_params = sum(p.numel() for p in self.model.age_head.parameters())
        
        self.logger.info(f"Parameter breakdown:")
        self.logger.info(f"  Total: {total_params:,}")
        self.logger.info(f"  Encoder: {encoder_params:,} ({'frozen' if freeze_encoder else 'trainable'})")
        self.logger.info(f"  Seg Head: {seg_head_params:,} (frozen)")
        self.logger.info(f"  Age Head: {age_head_params:,} (trainable)")
        self.logger.info(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        self.logger.info(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    def _step(self, batch: Dict[str, torch.Tensor], train: bool = True):
        imgs = batch["image"].to(self.device)
        age_gts = batch["age"].float().to(self.device)
        seg_gts = batch["seg_gt"].to(self.device)

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            seg_logits, age_preds = self.model(imgs)
            
            # Always compute age loss
            age_loss = self.age_criterion(age_preds, age_gts)
            
            # Check if we're in age-only mode
            if hasattr(self, 'age_only_mode') and self.age_only_mode:
                # Age-only training: just use age loss directly
                total_loss = age_loss
                seg_loss = torch.tensor(0.0, device=self.device)  # For logging only
            else:
                # Normal multitask training with uncertainty weighting
                seg_loss = self.seg_criterion(seg_logits, seg_gts)
                loss_age_weighted = torch.exp(-self.log_var_age) * age_loss + 0.5 * self.log_var_age
                loss_seg_weighted = torch.exp(-self.log_var_seg) * seg_loss + 0.5 * self.log_var_seg
                total_loss = loss_age_weighted.squeeze() + loss_seg_weighted.squeeze()

        if train:
            loss_acc = total_loss / self.grad_accum_steps
            if self.scaler:
                self.scaler.scale(loss_acc).backward()
            else:
                loss_acc.backward()
        
        # Still compute segmentation metrics for monitoring (even in age-only mode)
        seg_preds_for_metric = torch.argmax(seg_logits, dim=1, keepdim=True)
        self.dice_metric(y_pred=seg_preds_for_metric, y=seg_gts)

        return {
            "loss": total_loss.detach(), 
            "age_loss": age_loss.detach(), 
            "seg_loss": seg_loss.detach(),
            "age_pred": age_preds.detach(), 
            "age_gt": age_gts.detach(),
        }

    def _optim_step(self):
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _run_epoch(self, epoch: int, loader: DataLoader, train: bool):
        self.model.train(train)
        desc = "Train" if train else "Val/Test"
        
        epoch_losses = {"loss": 0.0, "age_loss": 0.0, "seg_loss": 0.0}
        age_preds_all, age_targets_all = [], []

        if train: self.optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} [{desc}]", leave=False)
        for i, batch in enumerate(pbar):
            with torch.set_grad_enabled(train):
                step_results = self._step(batch, train=train)

            for k in epoch_losses: epoch_losses[k] += step_results[k].item()
            age_preds_all.append(step_results["age_pred"].cpu().numpy())
            age_targets_all.append(step_results["age_gt"].cpu().numpy())

            if train and (i + 1) % self.grad_accum_steps == 0: self._optim_step()
            pbar.set_postfix({k: f"{v.item():.4f}" for k, v in step_results.items() if 'loss' in k})
        
        if train and (len(loader) % self.grad_accum_steps != 0): self._optim_step()
        if train and self.scheduler: self.scheduler.step()

        for k in epoch_losses: epoch_losses[k] /= len(loader)
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        age_preds_all = np.concatenate(age_preds_all).squeeze().astype(np.float64)
        age_targets_all = np.concatenate(age_targets_all).astype(np.float64)
        
        # # Add statistics logging
        # stage = "TRAIN" if train else "VAL"
        # self.logger.info(f"{stage} SUMMARY - Pred: min={age_preds_all.min():.2f}, max={age_preds_all.max():.2f}, mean={age_preds_all.mean():.2f}, std={age_preds_all.std():.2f}")
        # self.logger.info(f"{stage} SUMMARY - True: min={age_targets_all.min():.2f}, max={age_targets_all.max():.2f}, mean={age_targets_all.mean():.2f}, std={age_targets_all.std():.2f}")
        
        mae = np.mean(np.abs(age_preds_all - age_targets_all))
        
        metrics = {**epoch_losses, "mae": mae, "dice": dice_score}
        return metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {"epoch": epoch, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}
        if self.scheduler: ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        if is_best:
            fname = self.ckpt_dir / f"{self.exp_name}_best_mae.pt"
            torch.save(ckpt, fname)
            self.best_mae_checkpoint_path = str(fname)
            self.logger.info(f"Saved best MAE checkpoint: {fname}")

    def train(self) -> Dict[str, Any]:
        history = {k: [] for k in ("train_loss", "val_loss", "train_mae", "val_mae", "train_dice", "val_dice", "lr")}
        
        for epoch in range(self.epochs):
            train_metrics = self._run_epoch(epoch, self.train_loader, train=True)
            val_metrics = self._run_epoch(epoch, self.val_loader, train=False)

            for M in (('train', train_metrics), ('val', val_metrics)):
                for k, v in M[1].items(): history[f"{M[0]}_{k.replace('_loss','')}"] = v
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            log_dict = {f"train/{k}": v for k,v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k,v in val_metrics.items()})
            log_dict["lr"] = self.optimizer.param_groups[0]['lr']
            if self.use_wandb: self.wandb.log(log_dict, step=epoch+1)
            
            self.logger.info(f"Epoch {epoch+1}: Train MAE={train_metrics['mae']:.3f}, Dice={train_metrics['dice']:.3f} | Val MAE={val_metrics['mae']:.3f}, Dice={val_metrics['dice']:.3f}")

            is_best = val_metrics["mae"] < self.best_val_mae
            if is_best:
                self.best_val_mae = val_metrics["mae"]
                self.best_mae_epoch = epoch
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            self._save_checkpoint(epoch, is_best=is_best)

            if self.early_stop_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {"history": history, "best_mae_info": {"value": self.best_val_mae, "epoch": self.best_mae_epoch, "checkpoint_path": self.best_mae_checkpoint_path}}

    def evaluate(self, test_loader: DataLoader, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        if checkpoint_path:
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.model.eval()
        self.logger.info(f"Evaluating on {len(test_loader.dataset)} samples")
        metrics = self._run_epoch(-1, test_loader, train=False)
        self.logger.info(f"Evaluation results | MAE={metrics['mae']:.3f} Dice={metrics['dice']:.3f}")
        return metrics 