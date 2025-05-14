#!/usr/bin/env python
"""
Single-entry script that reads CSVs, builds the data-pipeline,

and launches training with weighted sampling + GPU transforms.
"""
import os, sys, time, json, random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
import wandb

# ───────────────────── project imports ────────────────────── #
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dom_rand.dataset import BADataset            # ← renamed module
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer
from brain_age_pred.models.sfcn import SFCN
from brain_age_pred.models.resnet3d import ResNet3D
from brain_age_pred.models.efficientnet3d import EfficientNet3D
from brain_age_pred.training.trainer import BrainAgeTrainer
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv, load_checkpoint
from torch.utils.data import WeightedRandomSampler
from dotenv import load_dotenv

load_dotenv()


def main() -> None:

    # 1. ─── configuration & reproducibility ─────────────────── #
    # Setup logger first so we can use it right away
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    cfg      = Config(cfg_file)
    
    # 2. ─── experiment naming / I/O ─────────────────────────── #
    experiment_name = cfg.get("output.experiment_name")
    if not experiment_name:
        experiment_name = f'{cfg.get("model.type","sfcn")}_{timestamp}'
    out_root  = Path(cfg.get("output.output_dir", "output"))
    ckpt_dir  = out_root / "checkpoints" / experiment_name
    log_dir   = out_root / "logs"        / experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("brain-age", log_file=log_dir / "train.log")
    
    logger.info("Initializing configuration...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Experiment: {experiment_name}\nConfig   : {cfg_file}")

    # 3. ─── W&B init ─────────────────────────────────────────── #
    logger.info("Initializing Weights & Biases...")
    use_wandb = cfg.get("wandb.use_wandb", True)
    if use_wandb:
        logger.info("Setting up W&B tracking")
        wandb.login(key=os.environ["WANDB_API"])
        wandb.init(
            project = cfg.get("wandb.project", "brain-age-pred"),
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

    # 5. ─── transforms (GPU-ready) ───────────────────────────── #
    logger.info("Initializing domain randomization transforms...")
    rand_cfg = cfg.get("domain_randomization", {})
    transform = DomainRandomizer(
        **rand_cfg,
        device=device,
    )
    logger.info("Domain randomizer initialized")

    # 6. ─── CSV → dataset / sampler ─────────────────────────── #
    logger.info("Reading CSV files...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv   = Path(cfg.get("data.val_csv"))
    test_csv  = Path(cfg.get("data.test_csv"))
    data_dir  = Path(cfg.get("data.data_dir"))

    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w = read_csv(
        train_csv,
        data_dir,
    )
    logger.info(f"Reading validation CSV from {val_csv}")
    val_p,   val_a, _ = read_csv(
        val_csv,
        data_dir,
    )
    logger.info(f"Reading test CSV from {test_csv}")
    test_p,  test_a, _ = read_csv(
        test_csv,
        data_dir,
    )


    logger.info("Initializing datasets...")
    logger.info("Creating training dataset")

    train_ds = BADataset(
        file_paths   = train_p,
        age_labels   = train_a,
        sample_wts   = train_w,
        transform    = transform,
        mode         = "train",
    )
    logger.info("Creating validation dataset")
    val_ds   = BADataset(
        file_paths   = val_p,
        age_labels   = val_a,
        transform    = None,
        mode         = "val",
    )

    test_ds = BADataset(

        file_paths   = test_p,
        age_labels   = test_a,
        transform    = None,
        mode         = "test",
    )

    logger.info("Setting up sampler...")

    sampler = WeightedRandomSampler(
        weights=train_w,
        num_samples=len(train_w),
        replacement=True,
    )
    logger.info("Weighted random sampler initialized")

    logger.info("Setting up data loader parameters...")
    dl_kwargs = dict(
        num_workers       = cfg.get("data.num_workers", 8),
        pin_memory        = (device.type == "cuda"),
        persistent_workers= cfg.get("data.persistent_workers", True),
        prefetch_factor   = cfg.get("data.prefetch_factor", 2),
    )

    logger.info("Creating training data loader")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size = cfg.get("training.batch_size", 8),
        sampler    = sampler,
        **dl_kwargs,
    )

    logger.info("Creating validation data loader")
    val_loader   = torch.utils.data.DataLoader(
        val_ds,
        batch_size = cfg.get("training.batch_size", 8),
        shuffle    = False,
        **dl_kwargs,
    )
    logger.info(f"Train={len(train_ds)}  Val={len(val_ds)}")

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size = cfg.get("training.batch_size", 8),
        shuffle    = False,
        **dl_kwargs,
    )

    # 7. ─── model ─────────────────────────────────────────────── #
    logger.info("Initializing model...")
    mtype = cfg.get("model.type", "sfcn").lower()
    model_map = {"sfcn": SFCN, "resnet3d": ResNet3D, "efficientnet3d": EfficientNet3D}

    if mtype == "sfcn":
        logger.info("Creating SFCN model")
        model = SFCN(
            in_channels=cfg.get("model.in_channels"),
            dropout_rate=cfg.get("model.dropout_rate"),
        ).to(device)
    elif mtype == "resnet3d":
        logger.info("Creating ResNet3D model")
        model = ResNet3D(
            in_channels=cfg.get("model.in_channels"),
            dropout_rate=cfg.get("model.dropout_rate"),
            use_attention=cfg.get("model.use_attention", False),
        ).to(device)
    else:
        logger.info(f"Creating {mtype} model")
        model = model_map[mtype](**cfg.get("model")).to(device)

    # Load checkpoint if specified
    checkpoint_path = cfg.get("model.checkpoint")
    if checkpoint_path:
        try:
            checkpoint_info = load_checkpoint(model, checkpoint_path, device, logger)
            if checkpoint_info:
                logger.info(f"Loaded checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
                if checkpoint_info.get('best_metric'):
                    logger.info(f"Best metric from checkpoint: {checkpoint_info['best_metric']}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    if use_wandb: 
        logger.info("Setting up W&B model watching")
        wandb.watch(model, log="all", log_graph=False)

    # params = list(model.parameters())
    # logger.debug("Model parameters:", params)
    # logger.debug("Number of parameters:", sum(p.numel() for p in params))

    # 8. ─── trainer ──────────────────────────────────────────── #
    logger.info("Initializing trainer...")
    trainer = BrainAgeTrainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        test_loader    = test_loader,
        config         = cfg.get("training"),
        device         = device,
        checkpoint_dir = ckpt_dir,
        log_dir        = log_dir,
        use_wandb      = use_wandb,
        wandb_project  = cfg.get("wandb.project", "brain-age-pred"),
        wandb_entity   = cfg.get("wandb.entity"),
        wandb_config   = cfg.config,
        experiment_name= experiment_name,
    )
    logger.info("Trainer initialized")

    # 9. ─── train ────────────────────────────────────────────── #
    logger.info("Starting training...")
    try:
        t0 = time.time()
        logger.info("Beginning training loop")
        history = trainer.train()
        logger.info(f"Training finished in {time.time()-t0:.1f}s")
        json.dump(history, open(ckpt_dir/"history.json","w"), indent=2)
        if use_wandb: wandb.log({"train/duration_s": time.time()-t0})
    except Exception as e:
        logger.error(f"Training failed: {e}")

    # 10. ─── evaluate ─────────────────────────────────────── #
    try:
        logger.info("Evaluating model...")
        metrics = trainer.evaluate(test_loader)
        logger.info(f"Evaluation results: {metrics}")
        if use_wandb: wandb.log({"test/metrics": metrics})
    except Exception as e:
        logger.error(f"Eval failed: {e}")
    finally:
        if use_wandb: wandb.finish()

    logger.info("All done.")

if __name__ == "__main__":
    main()