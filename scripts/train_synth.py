#!/usr/bin/env python
"""
Single-entry script that reads CSVs, builds the data-pipeline,

and launches training with weighted sampling + GPU transforms.
"""
import os, sys, time, json, random
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
from brain_age_pred.dataset.augmentation import AugmentationPipeline
from brain_age_pred.models.sfcn import SFCN
from brain_age_pred.models.sfcn_class import SFCNClass
from brain_age_pred.models.brainagenext import BrainAgeNeXt
from brain_age_pred.training.trainer import BrainAgeTrainer
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
        WANDB_API = '2abdb867a9244072f2237704a3cacc77fa548dd8'
        wandb.login(key=WANDB_API)
        wandb.init(
            project = cfg.get("wandb.project", "brain-age-synth"),
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
    
    # Read brain generator config from file
    bg_cfg = cfg.get("brain_generator", {})
    
    # Augmentation probabilities from config
    prob = bg_cfg.get("prob", {
        "flip": 0.5,
        "affine": 0.0,
        "contrast": 0.3,
        "gamma": 0.3,
        "scale_int": 0.3,
        "shift_int": 0.3,
        "hist_shift": 0.3,
        "noise": 0.3,
        "rician": 0.3,
        "gibbs": 0.3,
        "blur": 0.3,
        "bias": 0.0,
        "resolution": 0.3,
    })
    
    n_classes = GENERATION_CLASSES.max() + 1     # = 15 with the default label set

    # Prior distribution parameters
    mean_loc = bg_cfg.get("mean_loc", 125.0)
    mean_scale = bg_cfg.get("mean_scale", 100.0)
    std_loc = bg_cfg.get("std_loc", 15.0)
    std_scale = bg_cfg.get("std_scale", 10.0)
    
    n_classes = len(GENERATION_LABELS)

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

    # Update the BABrainGenerator initialization in train_synth.py
    brain_generator = BABrainGenerator(
        # Required parameters
        prior_means  = prior_means,
        prior_stds   = prior_stds,
        distribution = bg_cfg.get("distribution", "normal"),
        prob         = prob,

        # Spatial augmentation parameters
        rotation_range     = bg_cfg.get("rotation_range", 10),
        scaling_range      = bg_cfg.get("scaling_range", 0.1),
        shear_bounds       = bg_cfg.get("shear_bounds", 0.005),
        translation_bounds = bg_cfg.get("translation_bounds", False),

        # Intensity augmentation parameters
        contrast_range      = tuple(bg_cfg.get("contrast_range", [0.8, 1.2])),
        log_gamma_std       = bg_cfg.get("log_gamma_std", 0.1),
        shift_offset        = bg_cfg.get("shift_offset", 0.1),
        hist_control_points = bg_cfg.get("hist_control_points", 5),

        # Artefacts parameters
        noise_mean    = bg_cfg.get("noise_mean", 0.02),
        noise_std     = bg_cfg.get("noise_std", 0.015),
        rician_std    = bg_cfg.get("rician_std", 0.01),
        gibbs_alpha   = bg_cfg.get("gibbs_alpha", 0.4),
        blur_sigma    = bg_cfg.get("blur_sigma", 0.25),
        bias_field_rng= tuple(bg_cfg.get("bias_field_rng", [0.0, 0.5])),
        
        # Motion artifacts
        motion_degrees = bg_cfg.get("motion_degrees", 3),
        motion_translation = bg_cfg.get("motion_translation", 5),
        motion_num_transforms = bg_cfg.get("motion_num_transforms", 4),
        ghost_num = tuple(bg_cfg.get("ghost_num", [1, 4])),
        ghost_intensity = tuple(bg_cfg.get("ghost_intensity", [0.1, 0.6])),
        torchio_noise_std = bg_cfg.get("torchio_noise_std", [0, 0.5]),

        # Resolution parameters
        min_res       = bg_cfg.get("min_res", 0.8),
        max_res_iso   = bg_cfg.get("max_res_iso", 2.0),
        max_res_aniso = bg_cfg.get("max_res_aniso", 2.0),
        atlas_res     = bg_cfg.get("atlas_res", 1.0),
        thickness     = bg_cfg.get("thickness", None),

        # SynthSeg label config parameters
        generation_labels = GENERATION_LABELS,
        n_neutral_labels  = N_NEUTRAL_LABELS,
        output_labels     = None,

        # Toggle parameters
        use_sample                    = bg_cfg.get("use_sample", True),
        use_hemisphere_aware_flip     = bg_cfg.get("use_hemisphere_aware_flip", True),
        use_dynamic_resolution        = bg_cfg.get("use_dynamic_resolution", True),
        use_intensity_clip_normalize  = bg_cfg.get("use_intensity_clip_normalize", True),
        n_channels                    = bg_cfg.get("n_channels", 1),
        use_specific_stats_for_channel= bg_cfg.get("use_specific_stats_for_channel", False),
        output_shape = tuple(bg_cfg.get("output_shape", [182, 218, 182])),
        use_random_cropping          = bg_cfg.get("use_random_cropping", True),
        return_gradients             = bg_cfg.get("return_gradients", False),
        return_segmentation          = bg_cfg.get("return_segmentation", False),
        device                       = device,
    )
    print(f"Brain Generator config: {bg_cfg}")
            

    # 6. ─── CSV → dataset / sampler ─────────────────────────── #
    logger.info("Reading CSV files...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv   = Path(cfg.get("data.val_csv"))
    test_csv  = Path(cfg.get("data.test_csv"))
    segmented_data_dir = Path(cfg.get("data.segmented_data_dir"))
    real_data_dir  = Path(cfg.get("data.real_data_dir"))
    

    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w, train_s, train_m = read_csv(
        train_csv,
        segmented_data_dir,
    )
    logger.info(f"Reading val CSV from {val_csv}")

    val_p, val_a, val_w, val_s, val_m = read_csv(
        val_csv,
        real_data_dir,
    )
    logger.info(f"Reading test CSV from {test_csv}")

    test_p, test_a, test_w, test_s, test_m = read_csv(
        test_csv,
        real_data_dir,
    )
    
    logger.info(f"Train={len(train_p)}  Val={len(val_p)}  Test={len(test_p)}")
    logger.info(f"Sample weights from train: {train_w[0:10]}")

    # Add this right after reading CSVs
    print("=== AGE RANGES  ===")
    print(f"Train ages: min={min(train_a):.2f}, max={max(train_a):.2f}, mean={np.mean(train_a):.2f}, std={np.std(train_a):.2f}")
    print(f"Val ages:   min={min(val_a):.2f}, max={max(val_a):.2f}, mean={np.mean(val_a):.2f}, std={np.std(val_a):.2f}")
    print(f"Test ages:  min={min(test_a):.2f}, max={max(test_a):.2f}, mean={np.mean(test_a):.2f}, std={np.std(test_a):.2f}")
    print("Sample train ages:", train_a[:50])
    print("Sample val ages:", val_a[:50])
    print("Sample test ages:", test_a[:50])

    logger.info("Initializing datasets...")
    logger.info("Creating training dataset")

    train_ds = BADataset(
        file_paths   = train_p,
        age_labels   = train_a,
        sample_wts   = train_w,
        sexes        = train_s,
        modalities   = train_m,
        transform    = brain_generator,
        mode         = "train",
        cache_size   = cfg.get("data.cache_size", 0),
    )
    
    logger.info("Creating validation dataset")
    
    
    val_ds = BADataset(
        file_paths   = val_p,
        age_labels   = val_a,
        sexes        = val_s,
        modalities   = val_m,
        transform    = None,
        mode         = "val",
        cache_size   = cfg.get("data.cache_size", 0),
    )

    logger.info("Creating test dataset")
    test_ds = BADataset(
        file_paths   = test_p,
        age_labels   = test_a,
        sexes        = test_s,
        modalities   = test_m,
        transform    = None,
        mode         = "test",
        cache_size   = cfg.get("data.cache_size", 0),
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
        num_workers       = cfg.get("data.num_workers", 6),
        pin_memory        = cfg.get("data.pin_memory", True),
        pin_memory_device = cfg.get("data.pin_memory_device"),
        persistent_workers= cfg.get("data.persistent_workers", True),
        prefetch_factor   = cfg.get("data.prefetch_factor"),
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
    if mtype == "sfcn":
        logger.info("Creating SFCN model")
        model = SFCN(
            in_channels=cfg.get("model.in_channels"),
            dropout_rate=cfg.get("model.dropout_rate"),
            age_min=cfg.get("age_min"),
            age_max=cfg.get("age_max"),
        ).to(device)
    elif mtype == "sfcn_class":
        logger.info("Creating SFCN Class model")
        model = SFCNClass(
            in_channels=cfg.get("model.in_channels"),
            dropout_rate=cfg.get("model.dropout_rate"),
            channels=cfg.get("model.channels", (32, 64, 128, 256, 256, 64)),
            age_min=cfg.get("data.age_min"),
            age_max=cfg.get("data.age_max"),
        ).to(device)
    elif mtype == "brainagenext":
        logger.info("Creating BrainAgeNext model...")
        model = BrainAgeNeXt(
            in_channels=cfg.get("model.in_channels"),
            dropout_rate=cfg.get("model.dropout_rate"),
            model_id=cfg.get("model.model_id", "B"),
            kernel_size=cfg.get("model.kernel_size", 3),
            deep_supervision=cfg.get("model.deep_supervision", True),
            feature_size=cfg.get("model.feature_size", 512),
            hidden_size=cfg.get("model.hidden_size", 64),
        ).to(device)
    print(f"Model hyperparameters: {cfg.get('model')}")

    # Load checkpoint if specified
    checkpoint_path = cfg.get("model.checkpoint")
    if checkpoint_path:
        try:
            checkpoint_info = load_checkpoint(model, checkpoint_path, device, logger)
            if checkpoint_info:
                logger.info(f"Loaded checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    if use_wandb: 
        logger.info("Setting up W&B model watching")
        wandb.watch(model, log="all", log_graph=False)


    # 8. ─── trainer ──────────────────────────────────────────── #
    logger.info("Initializing trainer...")
    print(f"Trainer config: {cfg.get('training')}")
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
        age_min        = cfg.get("data.age_min"),
        age_max        = cfg.get("data.age_max"),
        wandb_project  = cfg.get("wandb.project", "brain-age-pred"),
        wandb_entity   = cfg.get("wandb.entity"),
        wandb_config   = cfg.config,
        experiment_name= experiment_name,)
    logger.info("Trainer initialized")

    # 9. ─── train ────────────────────────────────────────────── #
    logger.info("Starting training...")
    try:
        t0 = time.time()
        logger.info("Beginning training loop")
        
        # Updated: receive the enhanced return value from trainer.train()
        training_results = trainer.train()
        history = training_results["history"]
        best_mae_info = training_results["best_mae_info"]
        
        logger.info(f"Training finished in {time.time()-t0:.1f}s")
        json.dump(history, open(ckpt_dir/"history.json","w"), indent=2)
        json.dump(best_mae_info, open(ckpt_dir/"best_mae_info.json","w"), indent=2)
        
        if use_wandb: 
            wandb.log({"train/duration_s": time.time()-t0})
            wandb.log({
                "best_val_mae": best_mae_info["value"], 
                "best_val_mae_epoch": best_mae_info["epoch"] + 1
            })
        
        best_val = best_mae_info["value"]
        if np.isinf(best_val):                     
            raise Exception("Best validation MAE is not yet set")
        
    except Exception:
        logger.error(f"Training failed")
        raise


    try:
        logger.info("Starting 3-fold evaluation using best MAE checkpoint...")
        best_mae_checkpoint = best_mae_info["checkpoint_path"]
        logger.info(f"Loading best checkpoint from epoch {best_mae_info['epoch']+1} with MAE {best_mae_info['value']:.4f}")
        
        # 1. Normal test evaluation
        logger.info("=== Normal test evaluation ===")
        normal_metrics = trainer.evaluate(test_loader, checkpoint_path=best_mae_checkpoint)
        logger.info(f"Normal test results: {normal_metrics}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if use_wandb: wandb.finish()
        logger.info("All done.")
        return float(best_val)

if __name__ == "__main__":
    sys.exit(main())