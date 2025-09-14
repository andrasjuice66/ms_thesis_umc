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
import math
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
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
        use_wandb: bool = False,
        wandb_project: str = "brain-age",
        age_min: int = 20,
        age_max: int = 80,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
    ) -> None:

        # /--------- basic attributes ----------/
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
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
        self.best_val_loss = float("inf")
        self.best_val_mae = float("inf")  # Best validation MAE
        self.early_stop_counter = 0
        self.best_mae_epoch = -1  # Epoch with best validation MAE
        self.best_mae_checkpoint_path = None  # Path to best MAE checkpoint

        self.bin_step  = self.cfg.get("bin_step", 1)
        self.soft_sigma = self.cfg.get("loss_params", {}).get("sigma", 1.0)

        self.age_min = age_min
        self.age_max = age_max

        # ---------------------------------------------------
        # Integer-centred bins (match the SFCN constructor)
        # ---------------------------------------------------
        self.bin_centres = torch.arange(
            self.age_min,
            self.age_max + 1,           # inclusive
            self.bin_step,
            device=self.device,
            dtype=torch.float32,
        )
        self.n_bins = len(self.bin_centres)

        # /--------- move model ----------/
        self.model.to(self.device)
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        if train_loader is not None:
            self.logger.info(f"Training samples : {len(train_loader.dataset)}")
        if val_loader is not None:
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        if test_loader is not None:
            self.logger.info(f"Test samples: {len(test_loader.dataset)}")
        self.logger.info(f"Use AMP: {self.use_amp}")
        self.logger.info(f"Learning rate: {self.cfg.get('learning_rate', 1e-4)}")
        self.logger.info(f"Total model parameters: {sum(p.numel() for p in self.model.parameters())}")



    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Handles both KL-divergence (soft-labels) and classic regression losses.
        """
        if self.loss_name == "kl_div" or self.loss_name == "exp_mae":
            # outputs are log-probabilities, targets are probabilities
            return self.criterion(outputs, targets)

        # Ensure proper shape for regression
        outputs_squeezed = outputs.squeeze()
        if outputs_squeezed.dim() == 0 and targets.dim() > 0:
            outputs_squeezed = outputs_squeezed.unsqueeze(0)

        if weights is not None and self.loss_name in ["weighted_mse", "weighted_mae"]:
            return self.criterion(outputs_squeezed, targets, weights)
        else:
            return self.criterion(outputs_squeezed, targets)

    # ------------------------------------------------------------------ #
    def _step(self, batch: Dict[str, torch.Tensor], train: bool = True):
        imgs = batch["image"].to(self.device, non_blocking=True)
        ages = batch["age"].float().to(self.device, non_blocking=True)
        wts  = batch.get("weight")
        if wts is not None:
            wts = wts.to(self.device, non_blocking=True)

        # ─── forward (AMP or fp32) ───────────────────────────────── #
        def fwd():
            out = self.model(imgs)                       # logits or scalar
            
            if self.loss_name == "kl_div" or self.loss_name == "exp_mae":
                tgt   = self._soft_label(ages)
                loss  = self._compute_loss(out, tgt)     # KL
                pred  = (out.exp() * self.bin_centres.to(out.device)).sum(dim=1, keepdim=True)
            else:
                loss  = self._compute_loss(out, ages, wts)
                # Ensure proper shape for regression predictions
                pred  = out.squeeze()
                if pred.dim() == 0 and ages.dim() > 0:
                    pred = pred.unsqueeze(0)

            return loss, pred

        if self.use_amp:
            with autocast(device_type=self.device.type):
                loss, pred = fwd()
        else:
            loss, pred = fwd()

        # ─── backward / optim ────────────────────────────────────── #
        if train:
            loss_acc = loss / self.grad_accum_steps
            if self.use_amp:
                self.scaler.scale(loss_acc).backward()
            else:
                loss_acc.backward()
        return loss.detach(), pred.detach()


    def _soft_label(self, ages: torch.Tensor) -> torch.Tensor:
        if self.soft_sigma == 0:
            idx = torch.clamp(
                torch.floor((ages - self.age_min) / self.bin_step).long(),
                0, self.n_bins - 1,
            )
            return F.one_hot(idx, num_classes=self.n_bins).float()

        # integer centres → half-year boundaries
        left  = self.bin_centres.to(ages.device) - self.bin_step / 2
        right = self.bin_centres.to(ages.device) + self.bin_step / 2
        
        sqrt2 = math.sqrt(2.0)
        Φ = lambda x: 0.5 * (1 + torch.erf(x / (self.soft_sigma * sqrt2)))

        cdf_left  = Φ((left.unsqueeze(0) - ages.unsqueeze(1)))
        cdf_right = Φ((right.unsqueeze(0) - ages.unsqueeze(1)))
        probs = cdf_right - cdf_left                      # (B , n_bins)
        return probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    # ------------------------------------------------------------------ #
    def _optim_step(self) -> None:
        """Handles optimiser + scaler step for AMP."""

        # Add gradient clipping to prevent exploding gradients
        if self.use_amp:
            # Unscale gradients before clipping when using AMP
            self.scaler.unscale_(self.optimizer)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """
        One training epoch that works for *both*
        – regression models returning (B,) / (B,1)
        – soft-classification models returning log-probs (B, n_bins)

        All task-specific details are hidden inside `_step`.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running_loss   = 0.0                      # for average epoch loss
        data_time_tot  = 0.0
        gpu_time_tot   = 0.0

        preds_all, targets_all = [], []
        paths_all = []  # Track file paths for NaN debugging

        pbar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            leave=False,
            desc=f"Epoch {epoch+1}/{self.epochs} [train]",
        )

        for step, batch in enumerate(pbar):
            # --------------- host ➜ device time ------------------------ #
            t0 = time.perf_counter()
            batch = {k: v.to(self.device, non_blocking=True) if
                    torch.is_tensor(v) else v for k, v in batch.items()}
            data_time_tot += time.perf_counter() - t0

            # ------------------- forward / backward -------------------- #
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()

            loss, preds = self._step(batch, train=True)

            end_evt.record()
            torch.cuda.synchronize()
            gpu_time_tot += start_evt.elapsed_time(end_evt) / 1e3

            # --------------- grad accumulation ------------------------ #
            if (step + 1) % self.grad_accum_steps == 0:
                self._optim_step()

            # ---------------- bookkeeping ----------------------------- #
            running_loss += loss.item()
            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch["age"].cpu().numpy())
            
            # Collect file paths for NaN debugging
            if "__image_path__" in batch:
                batch_paths = batch["__image_path__"]
                if isinstance(batch_paths, (list, tuple)):
                    paths_all.extend(batch_paths)
                else:
                    paths_all.append(batch_paths)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # last, if #batches is not divisible by grad_accum_steps
        if (len(self.train_loader) % self.grad_accum_steps) != 0:
            self._optim_step()

        # ---------------- scheduler (per-epoch) ----------------------- #
        if self.scheduler is not None:
            self.scheduler.step()

        # ---------------- Metrics & logging -------------------------- #
        y_pred = np.concatenate(preds_all).astype(np.float64)
        y_true = np.concatenate(targets_all).astype(np.float64)

        # Check for NaN predictions and log the offending files
        if np.isnan(y_pred).any():
            nan_indices = np.where(np.isnan(y_pred))[0]
            self.logger.error(f"Found NaN predictions at indices: {nan_indices.tolist()}")
            
            # Log the file paths that caused NaN predictions
            if len(paths_all) >= len(y_pred):
                nan_files = [paths_all[i] for i in nan_indices if i < len(paths_all)]
                self.logger.error(f"Files that produced NaN predictions:")
                for i, file_path in enumerate(nan_files):
                    self.logger.error(f"  Index {nan_indices[i]}: {file_path}")
            else:
                self.logger.error(f"Could not map NaN indices to file paths (paths: {len(paths_all)}, predictions: {len(y_pred)})")

        # Log summary statistics
        self.logger.info(f"TRAIN SUMMARY - Pred: min={y_pred.min():.2f}, max={y_pred.max():.2f}, mean={y_pred.mean():.2f}, std={y_pred.std():.2f}")
        self.logger.info(f"TRAIN SUMMARY - True: min={y_true.min():.2f}, max={y_true.max():.2f}, mean={y_true.mean():.2f}, std={y_true.std():.2f}")

        metrics = calculate_metrics(y_pred, y_true)
        metrics.update({
            "loss"      : running_loss / len(self.train_loader),
            "data_time" : data_time_tot / len(self.train_loader),
            "gpu_time"  : gpu_time_tot  / len(self.train_loader),
        })

        self.logger.info(
            f"Epoch {epoch+1:03d} train | "
            f"loss={metrics['loss']:.4f}  mae={metrics['mae']:.3f}  "
            f"data={metrics['data_time']:.3f}s  gpu={metrics['gpu_time']:.3f}s"
        )

        return metrics

    # ------------------------------------------------------------------ #
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_loss: float = 0.0
        preds_all, targets_all = [], []
        modalities_all, sexes_all = [], []
        paths_all = []  # Track file paths for NaN debugging

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
                
                # Collect file paths for NaN debugging
                if "__image_path__" in batch:
                    batch_paths = batch["__image_path__"]
                    if isinstance(batch_paths, (list, tuple)):
                        paths_all.extend(batch_paths)
                    else:
                        paths_all.append(batch_paths)
                
                if "modality" in batch:
                    modalities_all.extend(batch["modality"])
                if "sex" in batch:
                    sexes_all.extend(batch["sex"])
                pbar.set_postfix(loss=loss.item())

        y_pred = np.concatenate(preds_all).astype(np.float64)
        y_true = np.concatenate(targets_all).astype(np.float64)
        
        # Check for NaN predictions and log the offending files
        if np.isnan(y_pred).any():
            nan_indices = np.where(np.isnan(y_pred))[0]
            self.logger.error(f"Found NaN predictions in validation at indices: {nan_indices.tolist()}")
            
            # Log the file paths that caused NaN predictions
            if len(paths_all) >= len(y_pred):
                nan_files = [paths_all[i] for i in nan_indices if i < len(paths_all)]
                self.logger.error(f"Validation files that produced NaN predictions:")
                for i, file_path in enumerate(nan_files):
                    self.logger.error(f"  Index {nan_indices[i]}: {file_path}")
            else:
                self.logger.error(f"Could not map NaN indices to file paths (paths: {len(paths_all)}, predictions: {len(y_pred)})")
        
        # Log summary statistics
        self.logger.info(f"VAL SUMMARY - Pred: min={y_pred.min():.2f}, max={y_pred.max():.2f}, mean={y_pred.mean():.2f}, std={y_pred.std():.2f}")
        self.logger.info(f"VAL SUMMARY - True: min={y_true.min():.2f}, max={y_true.max():.2f}, mean={y_true.mean():.2f}, std={y_true.std():.2f}")

        metrics = calculate_metrics(
            y_pred,
            y_true,
            modalities=modalities_all if modalities_all else None,
            sexes=sexes_all if sexes_all else None,
        )
        metrics["loss"] = running_loss / len(self.val_loader)

        self.logger.info(
            f"Epoch {epoch+1:03d}  val   | "
            f"loss={metrics['loss']:.4f}  mae={metrics['mae']:.3f}"
        )

        return metrics

    # ------------------------------------------------------------------ #
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_mae: float = None,
        is_best_loss: bool = False,
        is_best_mae: bool = False,
    ) -> None:
        """
        Save model checkpoint with options for best loss and best MAE versions.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        val_loss : float
            Validation loss value
        val_mae : float, optional
            Validation MAE value
        is_best_loss : bool, default=False
            Whether this checkpoint has the best validation loss so far
        is_best_mae : bool, default=False
            Whether this checkpoint has the best validation MAE so far
        """
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_loss": val_loss,
            "val_mae": val_mae,
        }
        
        # Save regular epoch checkpoint
        fname = self.ckpt_dir / f"{self.exp_name}_epoch{epoch:03d}.pt"
        torch.save(ckpt, fname)
        
        # Save best loss checkpoint
        if is_best_loss:
            best_loss_name = self.ckpt_dir / f"{self.exp_name}_best_loss.pt"
            torch.save(ckpt, best_loss_name)
        
        # Save best MAE checkpoint
        if is_best_mae:
            best_mae_name = self.ckpt_dir / f"{self.exp_name}_best_mae.pt"
            torch.save(ckpt, best_mae_name)
            self.best_mae_checkpoint_path = str(best_mae_name)
            self.logger.info(f"Saved best MAE checkpoint: {self.best_mae_checkpoint_path}")

    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, Any]:
        """
        Full training loop.  Returns history dict that contains:
            train_loss, val_loss, train_mae, val_mae, learning_rate
        
        Also returns information about the best model based on validation MAE.
        """
        history = {k: [] for k in
                   ("train_loss", "val_loss", "train_mae", "val_mae", "lr")}

        for epoch in range(self.epochs):
            # This line already exists and will now trigger probability updates
            if hasattr(self.train_loader.dataset.transform, "current_epoch"):
                self.train_loader.dataset.transform.current_epoch = epoch
                self.logger.info(f"Progressive dom rand update for epoch: {epoch+1}/{self.epochs}")

            tr_metrics = self.train_epoch(epoch)
            vl_metrics = self.validate(epoch)

            history["train_loss"].append(tr_metrics["loss"])
            history["val_loss"].append(vl_metrics["loss"])
            history["train_mae"].append(tr_metrics["mae"])
            history["val_mae"].append(vl_metrics["mae"])
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Log everything at once to wandb
            if self.use_wandb:
                log_dict = {f"train/{k}": v for k, v in tr_metrics.items()}
                log_dict.update({f"val/{k}": v for k, v in vl_metrics.items()})
                log_dict["lr"] = self.optimizer.param_groups[0]["lr"]
                self.wandb.log(log_dict, step=epoch+1)

            # Check for best validation MAE (this is the metric we care about)
            is_best_mae = vl_metrics["mae"] < self.best_val_mae
            if is_best_mae:
                self.best_val_mae = vl_metrics["mae"]
                self.best_mae_epoch = epoch
                self.logger.info(f"New best validation MAE: {self.best_val_mae:.4f} at epoch {epoch+1}")

            # Early stopping should use MAE, not loss
            self.early_stop_counter = 0 if is_best_mae else self.early_stop_counter + 1
            
            # Save checkpoint with both loss and MAE info
            self._save_checkpoint(
                epoch, 
                vl_metrics["loss"],
                vl_metrics["mae"],
                is_best_loss=False,
                is_best_mae=is_best_mae
            )

            if self.early_stop_counter >= self.early_stopping_patience:
                self.logger.info(
                    f"Early-stopping triggered at epoch {epoch+1}"
                )
                break

        # Log summary information about best MAE
        self.logger.info(f"Training completed")
        self.logger.info(f"Best validation MAE: {self.best_val_mae:.4f} achieved at epoch {self.best_mae_epoch+1}")
        
        # Return both the history and information about the best model
        return {
            "history": history,
            "best_mae_info": {
                "value": self.best_val_mae,
                "epoch": self.best_mae_epoch,
                "checkpoint_path": self.best_mae_checkpoint_path
            }
        }

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a test set.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader for the test set
        checkpoint_path : Optional[str]
            Path to checkpoint to load. If None, uses current model state.

        Returns
        -------
        Dict[str, float]
            Dictionary with evaluation metrics
        """
        if checkpoint_path:
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        self.logger.info(f"Evaluating on {len(test_loader.dataset)} samples")

        running_loss: float = 0.0
        preds_all, targets_all = [], []
        modalities_all, sexes_all = [], []

        with torch.no_grad():
            pbar = tqdm(
                test_loader,
                total=len(test_loader),
                leave=False,
                desc="Evaluation",
            )
            for batch in pbar:
                loss, preds = self._step(batch, train=False)
                running_loss += loss.item()
                preds_all.append(preds.cpu().numpy())
                targets_all.append(batch["age"].cpu().numpy())
                if "modality" in batch:
                    modalities_all.extend(batch["modality"])
                if "sex" in batch:
                    sexes_all.extend(batch["sex"])
                pbar.set_postfix(loss=loss.item())

        metrics = calculate_metrics(
            np.concatenate(preds_all),
            np.concatenate(targets_all),
            modalities=modalities_all if modalities_all else None,
            sexes=sexes_all if sexes_all else None,
        )
        metrics["loss"] = running_loss / len(test_loader)

        self.logger.info(
            f"Evaluation results | "
            f"loss={metrics['loss']:.4f}  mae={metrics['mae']:.3f}  "
            f"mse={metrics['mse']:.3f}  r2={metrics['r2']:.3f}"
        )

        if self.use_wandb:
            log_dict = {f"test/{k}": v for k, v in metrics.items()}
            self.wandb.log(log_dict)

        return metrics