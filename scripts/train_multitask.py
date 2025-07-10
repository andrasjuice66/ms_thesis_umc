#!/usr/bin/env python
"""
Multi-task training script for brain age prediction and segmentation.
"""
import os, sys, time, json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp

import pandas as pd
import numpy as np
import torch
import wandb

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dataset.dataset import BADataset
from brain_age_pred.models.multi_head import MultiTaskBrainAge
from brain_age_pred.training.trainer_multi_task import MultiTaskTrainer
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv, load_checkpoint
from torch.utils.data import WeightedRandomSampler, DataLoader
from brain_age_pred.brain_gen.brain_generator import BABrainGenerator
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES, GENERATION_LABELS, N_NEUTRAL_LABELS


def main() -> None:

    # 1. ─── configuration & reproducibility ─────────────────── #
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    cfg      = Config(cfg_file)
    
    # 2. ─── experiment naming / I/O ─────────────────────────── #
    experiment_name = cfg.get("output.experiment_name")
    if not experiment_name:
        experiment_name = f'multitask_{timestamp}'
    out_root  = Path(cfg.get("output.output_dir", "output"))
    ckpt_dir  = out_root / "checkpoints" / experiment_name
    log_dir   = out_root / "logs"        / experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("brain-age-multitask", log_file=log_dir / "train.log")
    
    logger.info("Initializing configuration...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Experiment: {experiment_name}\nConfig   : {cfg_file}")

    # 3. ─── W&B init ─────────────────────────────────────────── #
    logger.info("Initializing Weights & Biases...")
    use_wandb = cfg.get("wandb.use_wandb", True)
    if use_wandb:
        logger.info("Setting up W&B tracking")
        # Ensure you have your WANDB_API_KEY in your environment variables
        wandb.login(key="2abdb867a9244072f2237704a3cacc77fa548dd8")
        wandb.init(
            project = cfg.get("wandb.project", "brain-age-multitask"),
            entity  = cfg.get("wandb.entity"),
            name    = experiment_name,
            config  = cfg.config,
            reinit  = True,
        )
        cfg.save_config(ckpt_dir / "config.yaml")

    # 4. ─── device ───────────────────────────────────────────── #
    logger.info("Setting up device...")
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 5. ─── Brain Generator ──────────────────────────────────── #
    logger.info("Initializing Brain Generator...")
    bg_cfg = cfg.get("brain_generator", {})
    
    # Prior distribution parameters
    mean_loc = bg_cfg.get("mean_loc", 125.0)
    mean_scale = bg_cfg.get("mean_scale", 125.0)
    std_loc = bg_cfg.get("std_loc", 17.5)
    std_scale = bg_cfg.get("std_scale", 17.5)
    
    n_classes = GENERATION_CLASSES.max() + 1

    # "loc" = mid-point,  "scale" = half-range
    prior_means = np.vstack([
        np.full(n_classes, mean_loc,   dtype=float),
        np.full(n_classes, mean_scale, dtype=float),
    ])

    prior_stds = np.vstack([
        np.full(n_classes, std_loc,    dtype=float),
        np.full(n_classes, std_scale,  dtype=float),
    ])

    # Set background class (label 0) to zero
    prior_means[:, 0] = 0.0    
    prior_stds[:, 0] = 0.0     
    
    # ... existing code ...
    brain_generator = BABrainGenerator(
    # Critical for multi-task
    return_segmentation=True,

    # Pass other params from config
    prior_means=prior_means,
    prior_stds=prior_stds,
    distribution=bg_cfg.get("distribution", "normal"),
    prob=bg_cfg.get("prob", {}),
    
    # Spatial augmentation parameters
    rotation_range=bg_cfg.get("rotation_range", 10),
    scaling_range=bg_cfg.get("scaling_range", 0.1),
    shear_bounds=bg_cfg.get("shear_bounds", 0.005),
    translation_bounds=bg_cfg.get("translation_bounds", False),

    # Intensity augmentation parameters
    contrast_range=tuple(bg_cfg.get("contrast_range", [0.8, 1.2])),
    log_gamma_std=bg_cfg.get("log_gamma_std", 0.1),
    shift_offset=bg_cfg.get("shift_offset", 0.1),
    hist_control_points=bg_cfg.get("hist_control_points", 5),

    # ADD THESE MISSING PARAMETERS:
    # Artifact parameters
    noise_mean=bg_cfg.get("noise_mean", 0.5),
    noise_std=bg_cfg.get("noise_std", 0.08),
    rician_std=bg_cfg.get("rician_std", 0.08),
    gibbs_alpha=bg_cfg.get("gibbs_alpha", 0.5),  # Use middle value from typical range [0.0, 1.0]
    blur_sigma=bg_cfg.get("blur_sigma", 1.0),    # Use middle value from typical range [0.5, 2.0]
    bias_field_rng=tuple(bg_cfg.get("bias_field_rng", [0.0, 0.8])),
    
    # Resolution parameters
    min_res=bg_cfg.get("min_res", 1.0),
    max_res_iso=bg_cfg.get("max_res_iso", 1.8),

    output_shape=tuple(bg_cfg.get("output_shape", [160, 192, 160])),
    )
    print(f"Brain Generator config: {bg_cfg}")

    # 6. ─── CSV → dataset / sampler ─────────────────────────── #
    logger.info("Reading CSV files...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv   = Path(cfg.get("data.val_csv"))
    test_csv  = Path(cfg.get("data.test_csv"))
    segmented_data_dir = Path(cfg.get("data.segmented_data_dir"))

    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w, train_s, train_m = read_csv(train_csv, segmented_data_dir)
    logger.info(f"Reading val CSV from {val_csv}")
    val_p, val_a, val_w, val_s, val_m = read_csv(val_csv, segmented_data_dir)
    logger.info(f"Reading test CSV from {test_csv}")
    test_p, test_a, test_w, test_s, test_m = read_csv(test_csv, segmented_data_dir)

    logger.info(f"Train={len(train_p)}  Val={len(val_p)}  Test={len(test_p)}")

    logger.info("Initializing datasets...")
    train_ds = BADataset(train_p, train_a, train_w, train_s, train_m, transform=brain_generator, mode="train")
    # For multi-task, val and test also need the generator to get segmentation GT
    val_ds = BADataset(val_p, val_a, val_w, val_s, val_m, transform=brain_generator, mode="val")
    test_ds = BADataset(test_p, test_a, test_w, test_s, test_m, transform=brain_generator, mode="test")

    sampler = WeightedRandomSampler(weights=train_w, num_samples=len(train_w), replacement=True)
    dl_kwargs = dict(
        num_workers=cfg.get("data.num_workers", 6),
        pin_memory=cfg.get("data.pin_memory", True),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.get("training.batch_size", 8), sampler=sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("training.batch_size", 8), shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("training.batch_size", 8), shuffle=False, **dl_kwargs)

    # 7. ─── model ─────────────────────────────────────────────── #
    logger.info("Initializing MultiTaskBrainAge model...")
    model = MultiTaskBrainAge(n_classes=n_classes).to(device)
    print(f"Model hyperparameters: {cfg.get('model')}")

    checkpoint_path = cfg.get("model.checkpoint")
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, device, logger)
    
    if use_wandb:
        wandb.watch(model, log="all", log_graph=False)

    # 8. ─── trainer ──────────────────────────────────────────── #
    logger.info("Initializing MultiTaskTrainer...")
    print(f"Trainer config: {cfg.get('training')}")
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg.get("training"),
        device=device,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        use_wandb=use_wandb,
        wandb_project=cfg.get("wandb.project", "brain-age-multitask"),
        wandb_entity=cfg.get("wandb.entity"),
        wandb_config=cfg.config,
        experiment_name=experiment_name,
    )
    logger.info("Trainer initialized")

    # 9. ─── train ────────────────────────────────────────────── #
    logger.info("Starting multi-task training...")
    try:
        t0 = time.time()
        results = trainer.train()
        history = results["history"]
        best_mae_info = results["best_mae_info"]
        
        logger.info(f"Training finished in {time.time()-t0:.1f}s")
        json.dump(history, (ckpt_dir / "history.json").open("w"), indent=2)
        json.dump(best_mae_info, (ckpt_dir / "best_mae_info.json").open("w"), indent=2)
        
        if use_wandb:
            wandb.log({"train/duration_s": time.time() - t0})
            wandb.log({
                "best_val_mae": best_mae_info["value"],
                "best_val_mae_epoch": best_mae_info["epoch"] + 1
            })

        if np.isinf(best_mae_info["value"]):
            raise Exception("Best validation MAE was not set.")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    # 10. ─── evaluate ───────────────────────────────────────────── #
    try:
        logger.info("Starting evaluation using best MAE checkpoint...")
        best_mae_checkpoint = best_mae_info["checkpoint_path"]
        logger.info(f"Loading best checkpoint from epoch {best_mae_info['epoch']+1} with MAE {best_mae_info['value']:.4f}")
        
        test_metrics = trainer.evaluate(test_loader, checkpoint_path=best_mae_checkpoint)
        logger.info(f"Test results: {test_metrics}")
        if use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
    finally:
        if use_wandb:
            wandb.finish()
        logger.info("All done.")

if __name__ == "__main__":
    main() 