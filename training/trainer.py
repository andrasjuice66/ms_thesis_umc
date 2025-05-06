from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# ──────────────── project helpers ───────────────── #
from brain_age_pred.training.losses import get_loss_function
from brain_age_pred.training.metrics import calculate_metrics
from brain_age_pred.training.optimizers import get_optimizer, get_scheduler
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.wandb_logger import WandbLogger


# ---------------------------------------------------------------------------- #
#                              helper ‑ utilities                              #
# ---------------------------------------------------------------------------- #
def _weighted_reduction(
    per_sample: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce per-sample loss tensor with optional weights."""
    if weights is None:
        return per_sample.mean()
    w = weights / (weights.sum() + 1e-8)           # normalise
    return (per_sample * w).sum()


# ---------------------------------------------------------------------------- #
#                                 trainer                                      #
# ---------------------------------------------------------------------------- #
class BrainAgeTrainer:
    """
    Generic trainer that works with any **regression** model returning a tensor
    of shape (N,) or (N,1).

    Parameters
    ----------
    model          : torch.nn.Module
    train_loader   : DataLoader
    val_loader     : DataLoader
    config         : Dict    – subsection `training:` of YAML
    device         : torch.device
    checkpoint_dir : path to save checkpoints in
    log_dir        : path to save log file
    use_wandb      : enable Weights & Biases
    wandb_project / entity / config  : usual W&B params
    experiment_name: readable experiment id (used in filenames & wandb name)
    """

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
        use_wandb: bool = False,
        wandb_project: str = "brain-age",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
    ) -> None:

        # /--------- basic attributes ----------/
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = config
        self.device       = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # /--------- dirs & logging ---------/
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.exp_name = experiment_name or f"{model.__class__.__name__}_{int(time.time())}"
        self.logger   = setup_logger(
            name = self.exp_name,
            log_file = self.log_dir / f"{self.exp_name}.log"
        )

        # /--------- W&B ---------/
        self.use_wandb = use_wandb
        if use_wandb:
            self.wandb = WandbLogger(
                project = wandb_project,
                entity  = wandb_entity,
                name    = self.exp_name,
                config  = wandb_config or {},
            )

        # /--------- loss ---------/
        self.loss_name   = self.cfg.get("loss", "mse").lower()
        self.loss_params = self.cfg.get("loss_params", {})
        self.criterion   = get_loss_function(self.loss_name, **self.loss_params)

        # /--------- optimiser / scheduler ---------/
        params = list(self.model.parameters())
        # print("Modelllll parameters:", params)
        #print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))
        self.optimizer = get_optimizer(
            params,
            optimizer_type = self.cfg.get("optimizer", "adamw"),
            lr             = self.cfg.get("learning_rate", 1e-4),
            weight_decay   = self.cfg.get("weight_decay", 1e-5),
        )

        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type = self.cfg.get("scheduler", "cosine"),
            **self.cfg.get("scheduler_params", {}),
        )

        # /--------- misc hyper-params ---------/
        self.epochs                     = self.cfg.get("epochs", 100)
        self.grad_accum_steps           = self.cfg.get("gradient_accumulation_steps", 1)
        self.early_stopping_patience    = self.cfg.get("early_stopping_patience", 10)
        self.use_amp                    = self.cfg.get("use_amp", True) and torch.cuda.is_available()
        self.scaler: Optional[GradScaler] = GradScaler(device=self.device) if self.use_amp else None

        # /--------- early-stop bookkeeping ---------/
        self.best_val_loss      = float("inf")
        self.early_stop_counter = 0

        # /--------- move model ----------/
        self.model.to(self.device)
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples : {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"Use AMP: {self.use_amp}")

        # print("Model parameters:", list(model.parameters()))
        # print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # ------------------------------------------------------------------ #
    #                        internal helpers                             #
    # ------------------------------------------------------------------ #
    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns scalar loss; if `weighted_mse` is chosen or `weights`
        tensor is provided we override reduction manually.
        """
        if self.loss_name == "weighted_mse" or weights is not None:
            per_sample = F.mse_loss(
                outputs.squeeze(), targets, reduction="none"
            )
            return _weighted_reduction(per_sample, weights)
        # default: rely on criterion's internal reduction
        return self.criterion(outputs.squeeze(), targets)

    # ------------------------------------------------------------------ #
    def _step(
        self,
        batch: Dict[str, torch.Tensor],
        train: bool = True,
    ) -> torch.Tensor:
        """One forward/backward (if train) pass. Returns scalar loss."""
        # First ensure all tensors are on the correct device
        imgs = batch["image"].to(self.device, non_blocking=True)
        ages = batch["age"].float().to(self.device, non_blocking=True)
        wts = batch.get("weight")
        if wts is not None:
            wts = wts.to(self.device)

        # Forward pass (with AMP if enabled)
        if self.use_amp:
            with autocast(device_type=self.device.type):
                preds = self.model(imgs)
                loss = self._compute_loss(preds, ages, wts)
        else:
            preds = self.model(imgs)
            loss = self._compute_loss(preds, ages, wts)

        if train:
            loss = loss / self.grad_accum_steps
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss.detach(), preds.detach()

    # ------------------------------------------------------------------ #
    def _optim_step(self) -> None:
        """Handles optimiser + scaler step for AMP."""
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss: float = 0.0
        preds_all, targets_all = [], []
        
        # Add metrics for timing
        data_time_avg = 0.0
        gpu_time_avg = 0.0
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            leave=False,
            desc=f"Epoch {epoch+1}/{self.epochs} [train]",
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in pbar:
            # Start timing data loading
            t0 = time.time()
            
            # Move data to device (this is part of data pipeline timing)
            imgs = batch["image"].to(self.device, non_blocking=True)
            ages = batch["age"].float().to(self.device, non_blocking=True)
            wts = batch.get("weight")
            if wts is not None:
                wts = wts.to(self.device)
            
            # Calculate data loading time
            data_time = time.time() - t0
            data_time_avg += data_time
            
            # Ensure GPU operations from data loading are complete
            torch.cuda.synchronize()
            t1 = time.time()
            
            # Forward pass and backward pass (already implemented in _step)
            loss, preds = self._step(batch, train=True)
            
            # Apply optimizer step if needed
            if (step + 1) % self.grad_accum_steps == 0:
                self._optim_step()
            
            # Ensure GPU operations are complete
            torch.cuda.synchronize()
            gpu_time = time.time() - t1
            gpu_time_avg += gpu_time
            
            # Track metrics
            running_loss += loss.item() * self.grad_accum_steps
            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch["age"].cpu().numpy())
            
            # Update progress bar with both loss and timing info
            pbar.set_postfix(
                loss=loss.item() * self.grad_accum_steps,
                data_time=f"{data_time:.3f}s", 
                gpu_time=f"{gpu_time:.3f}s"
            )
            
            # Print timing for every step if needed (can be adjusted or removed)
            if step % 10 == 0:  # Print every 10 batches to avoid too much output
                self.logger.info(f"[{step:04d}] data {data_time:.3f}s | gpu {gpu_time:.3f}s")

        # Calculate average timings
        num_batches = len(self.train_loader)
        avg_data_time = data_time_avg / num_batches
        avg_gpu_time = gpu_time_avg / num_batches
        
        # Log timing statistics
        self.logger.info(f"Epoch {epoch+1} timing: avg data {avg_data_time:.3f}s | avg gpu {avg_gpu_time:.3f}s")
        
        # scheduler ‑ per-epoch step
        if self.scheduler is not None:
            self.scheduler.step()

        metrics = calculate_metrics(
            np.concatenate(preds_all),
            np.concatenate(targets_all),
        )
        metrics["loss"] = running_loss / num_batches
        # Add timing metrics
        metrics["data_time"] = avg_data_time
        metrics["gpu_time"] = avg_gpu_time

        # logging
        self.logger.info(
            f"Epoch {epoch+1:03d}  train | "
            f"loss={metrics['loss']:.4f}  mae={metrics['mae']:.3f} "
            f"data_time={avg_data_time:.3f}s  gpu_time={avg_gpu_time:.3f}s"
        )
        if self.use_wandb:
            self.wandb.log({f"train/{k}": v for k, v in metrics.items()})
            self.wandb.log({"lr": self.optimizer.param_groups[0]["lr"]})

        return metrics

    # ------------------------------------------------------------------ #
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_loss: float = 0.0
        preds_all, targets_all = [], []

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                total=len(self.val_loader),
                leave=False,
                desc=f"Epoch {epoch+1}/{self.epochs} [val]",
            )
            for batch in pbar:
                loss, preds = self._step(batch, train=False)
                running_loss += loss.item()
                preds_all.append(preds.cpu().numpy())
                targets_all.append(batch["age"].cpu().numpy())
                pbar.set_postfix(loss=loss.item())

        metrics = calculate_metrics(
            np.concatenate(preds_all),
            np.concatenate(targets_all),
        )
        metrics["loss"] = running_loss / len(self.val_loader)

        self.logger.info(
            f"Epoch {epoch+1:03d}  val   | "
            f"loss={metrics['loss']:.4f}  mae={metrics['mae']:.3f}"
        )
        if self.use_wandb:
            self.wandb.log({f"val/{k}": v for k, v in metrics.items()})

        return metrics

    # ------------------------------------------------------------------ #
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_loss": val_loss,
        }
        fname = self.ckpt_dir / f"{self.exp_name}_epoch{epoch:03d}.pt"
        torch.save(ckpt, fname)
        if is_best:
            best_name = self.ckpt_dir / f"{self.exp_name}_best.pt"
            torch.save(ckpt, best_name)

    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.  Returns history dict that contains:
            train_loss, val_loss, train_mae, val_mae, learning_rate
        """
        history = {k: [] for k in
                   ("train_loss", "val_loss", "train_mae", "val_mae", "lr")}

        for epoch in range(self.epochs):

            # let domain-randomiser know which epoch we are on
            if hasattr(self.train_loader.dataset.transform, "current_epoch"):
                self.train_loader.dataset.transform.current_epoch = epoch

            tr_metrics = self.train_epoch(epoch)
            vl_metrics = self.validate(epoch)

            history["train_loss"].append(tr_metrics["loss"])
            history["val_loss"].append(vl_metrics["loss"])
            history["train_mae"].append(tr_metrics["mae"])
            history["val_mae"].append(vl_metrics["mae"])
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # checkpoint & early-stopping
            is_best = vl_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = vl_metrics["loss"]
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            self._save_checkpoint(epoch, vl_metrics["loss"], is_best=is_best)

            if self.early_stop_counter >= self.early_stopping_patience:
                self.logger.info(
                    f"Early-stopping triggered at epoch {epoch+1}"
                )
                break

        if self.use_wandb:
            self.wandb.finish()

        return history