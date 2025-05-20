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
        test_loader: DataLoader,
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
        self.best_metric     = float("inf")       
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
        if self.loss_name == "weighted_mse" and weights is not None:
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
            wts = wts.to(self.device, non_blocking=True)

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
        """
        One full training epoch **without** unnecessary synchronisation
        or double device-transfer.  Returns a dict with loss/metrics +
        average data- and GPU-times (seconds).
        """
        self.model.train()

        running_loss = 0.0
        data_time_tot, gpu_time_tot = 0.0, 0.0
        preds_all, targets_all = [], []

        pbar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            leave=False,
            desc=f"Epoch {epoch+1}/{self.epochs} [train]",
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):

            # ─── Host  ➜  Device  ───────────────────────────────────────── #
            t0 = time.perf_counter()

            imgs = batch["image"].to(self.device, non_blocking=True)
            ages = batch["age"].to(self.device, non_blocking=True)
            wts  = batch.get("weight")
            if wts is not None:
                wts = wts.to(self.device, non_blocking=True)

            data_time_tot += time.perf_counter() - t0

            # ─── GPU compute  (timed with CUDA events)  ─────────────────── #
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()

            if self.use_amp:
                with autocast(device_type=self.device.type):
                    preds = self.model(imgs)
                    loss  = self._compute_loss(preds, ages, wts)
            else:
                preds = self.model(imgs)
                loss  = self._compute_loss(preds, ages, wts)

            loss = loss / self.grad_accum_steps
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            end_evt.record()
            torch.cuda.synchronize()                           # only for timing
            gpu_time_tot += start_evt.elapsed_time(end_evt) / 1e3  # → seconds

            # ─── bookkeeping & progress bar ────────────────────────────── #
            running_loss += loss.item() * self.grad_accum_steps
            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(ages.detach().cpu().numpy())

            pbar.set_postfix(
                loss=f"{loss.item()*self.grad_accum_steps:.4f}",
                data=f"{(data_time_tot/(step+1)):.3f}s",
                gpu=f"{(gpu_time_tot/(step+1)):.3f}s",
            )

        # ─── scheduler step (per-epoch) ─────────────────────────────────── #
        if self.scheduler is not None:
            self.scheduler.step()

        # ─── aggregate metrics ──────────────────────────────────────────── #
        num_batches   = len(self.train_loader)
        avg_data_time = data_time_tot / num_batches
        avg_gpu_time  = gpu_time_tot  / num_batches

        metrics = calculate_metrics(
            np.concatenate(preds_all),
            np.concatenate(targets_all),
        )
        metrics.update({
            "loss"      : running_loss / num_batches,
            "data_time" : avg_data_time,
            "gpu_time"  : avg_gpu_time,
        })

        # ─── logging ────────────────────────────────────────────────────── #
        self.logger.info(
            f"Epoch {epoch+1:03d} train | "
            f"loss={metrics['loss']:.4f}  mae={metrics['mae']:.3f}  "
            f"data={avg_data_time:.3f}s  gpu={avg_gpu_time:.3f}s"
        )

        return metrics
        # ------------------------------------------------------------------ #
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_loss: float = 0.0
        preds_all, targets_all = [], []
        modalities_all, sexes_all = [], []

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
            # This line already exists and will now trigger probability updates
            if hasattr(self.train_loader.dataset.transform, "current_epoch"):
                self.train_loader.dataset.transform.current_epoch = epoch

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

            # checkpoint & early-stopping
            is_best = vl_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = vl_metrics["loss"]

            if vl_metrics["mae"] < self.best_metric:   # <─ NEW
                self.best_metric = vl_metrics["mae"]   # <─ NEW

            self.early_stop_counter = 0 if is_best else self.early_stop_counter + 1
            self._save_checkpoint(epoch, vl_metrics["loss"], is_best=is_best)

            if self.early_stop_counter >= self.early_stopping_patience:
                self.logger.info(
                    f"Early-stopping triggered at epoch {epoch+1}"
                )
                break

        return history

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