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

        # ─── support for soft-classification SFCN ────────────────────── #
        # age grid & Gaussian-label bandwidth (σ) are configurable
        self.age_min   = self.cfg.get("age_min", 20)
        self.age_max   = self.cfg.get("age_max", 85)
        self.soft_sigma = self.cfg.get("loss_params", {}).get("sigma", 1.0)

        # tensor of bin centres, lives on the training device
        self.bin_centres = torch.arange(
            self.age_min,
            self.age_max + 1,
            device=self.device,
            dtype=torch.float32,
        )

        # /--------- DEBUG: Store initial weights for tracking changes ---------/
        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()

        # /--------- move model ----------/
        self.model.to(self.device)
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples : {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"Use AMP: {self.use_amp}")
        self.logger.info(f"Learning rate: {self.cfg.get('learning_rate', 1e-4)}")
        self.logger.info(f"Total model parameters: {sum(p.numel() for p in self.model.parameters())}")

        # /--------- DEBUG: Run initial sanity checks ---------/
        self.check_data_sanity()

        # /--------- DEBUG: Test single batch overfitting ---------/
        if not self.test_single_batch_overfitting():
            self.logger.error("CRITICAL: Model cannot overfit single batch!")
            raise RuntimeError("Model cannot learn - check architecture/data")

    # ------------------------------------------------------------------ #
    #                        DEBUG METHODS                               #
    # ------------------------------------------------------------------ #
    def check_data_sanity(self):
        """Check if data is reasonable"""
        self.logger.info("=== DATA SANITY CHECK ===")
        try:
            batch = next(iter(self.train_loader))
            imgs = batch["image"]
            ages = batch["age"]

            self.logger.info(f"Image shape: {imgs.shape}")
            self.logger.info(f"Image stats: min={imgs.min():.3f}, max={imgs.max():.3f}, mean={imgs.mean():.3f}, std={imgs.std():.3f}")
            self.logger.info(f"Ages: {ages}")
            self.logger.info(f"Age stats: min={ages.min():.1f}, max={ages.max():.1f}, mean={ages.mean():.1f}, std={ages.std():.1f}")

            # Check for NaN/inf
            if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                self.logger.error("CRITICAL: Images contain NaN/inf!")
                raise ValueError("Images contain NaN/inf")
            if torch.isnan(ages).any() or torch.isinf(ages).any():
                self.logger.error("CRITICAL: Ages contain NaN/inf!")
                raise ValueError("Ages contain NaN/inf")

            # Check if images are all zeros
            if imgs.abs().sum() == 0:
                self.logger.error("CRITICAL: All images are zeros!")
                raise ValueError("All images are zeros")

            self.logger.info("Data sanity check PASSED")
        except Exception as e:
            self.logger.error(f"Data sanity check FAILED: {e}")
            raise

    def test_single_batch_overfitting(self):
        """Test if model can overfit to a single batch"""
        self.logger.info("=== SINGLE BATCH OVERFITTING TEST ===")

        # Save current state
        original_state = self.model.state_dict()
        original_optimizer_state = self.optimizer.state_dict()

        try:
            self.model.train()

            # Get one batch
            batch = next(iter(self.train_loader))
            imgs = batch["image"].to(self.device)
            ages = batch["age"].float().to(self.device)

            self.logger.info(f"Testing with batch: images {imgs.shape}, ages {ages}")

            initial_loss = None
            for i in range(100):  # 100 steps on same batch
                self.optimizer.zero_grad()
                outputs = self.model(imgs)

                # Handle output shape
                if outputs.dim() > 1:
                    outputs = outputs.squeeze()
                if outputs.dim() == 0 and ages.dim() > 0:
                    outputs = outputs.unsqueeze(0)

                loss = F.mse_loss(outputs, ages)
                loss.backward()

                # Check gradients
                total_grad_norm = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5

                self.optimizer.step()

                if i == 0:
                    initial_loss = loss.item()
                    self.logger.info(f"Initial loss: {initial_loss:.6f}, grad_norm: {total_grad_norm:.6f}")

                if i % 20 == 0:
                    self.logger.info(f"Step {i}: Loss = {loss.item():.6f}, grad_norm: {total_grad_norm:.6f}")

                # Early exit if loss becomes very small
                if loss.item() < initial_loss * 0.01:
                    self.logger.info(f"Early convergence at step {i}")
                    break

            final_loss = loss.item()
            self.logger.info(f"Single batch test: Initial loss = {initial_loss:.6f}, Final loss = {final_loss:.6f}")
            self.logger.info(f"Loss reduction ratio: {final_loss/initial_loss:.6f}")

            success = final_loss < initial_loss * 0.1  # Should drop to <10% of initial
            if success:
                self.logger.info("Single batch overfitting test PASSED")
            else:
                self.logger.error("Single batch overfitting test FAILED - model cannot learn!")

        except Exception as e:
            self.logger.error(f"Single batch test failed with error: {e}")
            success = False
        finally:
            # Restore original state
            self.model.load_state_dict(original_state)
            self.optimizer.load_state_dict(original_optimizer_state)

        return success

    def check_weight_changes(self, epoch):
        """Check if model weights are actually changing"""
        total_change = 0
        max_change = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.initial_weights:
                change = (param.data - self.initial_weights[name]).abs().sum().item()
                total_change += change
                max_change = max(max_change, change)

        self.logger.info(f"Epoch {epoch}: Total weight change: {total_change:.6f}, Max change: {max_change:.6f}")

        if total_change < 1e-8:
            self.logger.warning("WARNING: Weights barely changing - possible learning issue!")

        return total_change

    def check_gradient_flow(self):
        """Check gradient magnitudes"""
        total_grad_norm = 0
        max_grad_norm = 0
        zero_grad_count = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                max_grad_norm = max(max_grad_norm, grad_norm)
            else:
                zero_grad_count += 1

        total_grad_norm = total_grad_norm ** 0.5

        self.logger.info(f"Gradient stats: total_norm={total_grad_norm:.6f}, max_norm={max_grad_norm:.6f}, zero_grads={zero_grad_count}")

        if total_grad_norm < 1e-6:
            self.logger.warning("WARNING: Gradients are extremely small - vanishing gradient problem!")
        if total_grad_norm > 100:
            self.logger.warning("WARNING: Gradients are very large - exploding gradient problem!")

        return total_grad_norm

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
        Handles both KL-divergence (soft-labels) and classic regression losses.
        """
        if self.loss_name == "kl_div":
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
            
            if self.loss_name == "kl_div":
                tgt   = self._soft_label(ages)
                loss  = self._compute_loss(out, tgt)     # KL
                pred  = (out.exp() * self.bin_centres).sum(dim=1, keepdim=True)
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
        """
        Build a Gaussian target distribution for each age.
        Returns shape (B, 66)  with rows summing to 1.
        """
        diff2 = (self.bin_centres - ages.unsqueeze(1)) ** 2
        g = torch.exp(-0.5 * diff2 / (self.soft_sigma ** 2))
        return g / (g.sum(dim=1, keepdim=True) + 1e-8)

    # ------------------------------------------------------------------ #
    def _optim_step(self) -> None:
        """Handles optimiser + scaler step for AMP."""
        # DEBUG: Check gradient flow before optimization
        grad_norm = self.check_gradient_flow()

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

            # Convert predictions and targets to numpy for logging
            preds_np = preds.cpu().numpy()
            targets_np = batch["age"].cpu().numpy()

            # Log detailed predictions for first few batches only
            if step < 3:  # Reduce logging spam
                for i in range(len(preds_np)):
                    pred_age = preds_np[i]
                    target_age = targets_np[i]
                    abs_error = abs(pred_age - target_age)
                    self.logger.info(f"Batch {step}, Sample {i} | Predicted: {pred_age:.2f} | Target: {target_age:.2f} | Error: {abs_error:.2f}")

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # last, if #batches is not divisible by grad_accum_steps
        if (len(self.train_loader) % self.grad_accum_steps) != 0:
            self._optim_step()

        # ---------------- scheduler (per-epoch) ----------------------- #
        if self.scheduler is not None:
            self.scheduler.step()

        # DEBUG: Check weight changes every 5 epochs
        if epoch % 5 == 0:
            self.check_weight_changes(epoch)

        # ---------------- Metrics & logging -------------------------- #
        y_pred = np.concatenate(preds_all)
        y_true = np.concatenate(targets_all)

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

        self.logger.info("Epoch Summary Statistics:")
        self.logger.info(f"Min prediction: {y_pred.min():.2f}")
        self.logger.info(f"Max prediction: {y_pred.max():.2f}")
        self.logger.info(f"Mean prediction: {y_pred.mean():.2f}")
        self.logger.info(f"Min target: {y_true.min():.2f}")
        self.logger.info(f"Max target: {y_true.max():.2f}")
        self.logger.info(f"Mean target: {y_true.mean():.2f}")
        self.logger.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")

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