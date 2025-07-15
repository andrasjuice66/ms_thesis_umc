#!/usr/bin/env python
"""
Multi-task training script with SynthSeg initialization for brain age prediction and segmentation.
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
from brain_age_pred.utils.weight_transfer import transfer_synthseg_weights
from torch.utils.data import WeightedRandomSampler, DataLoader
from brain_age_pred.brain_gen.brain_generator import BABrainGenerator
from brain_age_pred.brain_gen.validation_generator import ValidationGenerator
from brain_age_pred.brain_gen.labels import GENERATION_CLASSES, GENERATION_LABELS, N_NEUTRAL_LABELS


def load_synthseg_weights(model: torch.nn.Module, 
                         synthseg_path: str, 
                         freeze_encoder: bool = False, 
                         freeze_decoder: bool = False,
                         logger=None) -> dict:
    """
    Load SynthSeg weights into the model.
    
    Args:
        model: PyTorch model to load weights into
        synthseg_path: Path to SynthSeg .h5 file OR .pth file
        freeze_encoder: Whether to freeze encoder weights
        freeze_decoder: Whether to freeze decoder weights
        logger: Logger instance
    
    Returns:
        Dictionary with transfer summary
    """
    synthseg_path = Path(synthseg_path)
    if not synthseg_path.exists():
        raise FileNotFoundError(f"SynthSeg model not found: {synthseg_path}")
    
    if logger:
        logger.info(f"Loading SynthSeg weights from: {synthseg_path}")
    
    # Check file extension to determine loading method
    if synthseg_path.suffix == '.pth':
        # Load PyTorch checkpoint
        if logger:
            logger.info("Detected .pth file - loading as PyTorch checkpoint")
        
        checkpoint = torch.load(synthseg_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            transfer_summary = checkpoint.get('transfer_summary', {})
            model_config = checkpoint.get('model_config', {})
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint
            transfer_summary = {'transferred': {}, 'skipped': {}, 'transfer_stats': {'total_attempted': 0, 'successfully_transferred': 0, 'skipped_count': 0}}
        
        # Load the state dict
        model_dict = model.state_dict()
        transferred = {}
        skipped = {}
        
        for name, param in state_dict.items():
            if name in model_dict:
                if model_dict[name].shape == param.shape:
                    model_dict[name] = param
                    transferred[name] = param.shape
                else:
                    skipped[name] = f"Shape mismatch: expected {model_dict[name].shape}, got {param.shape}"
            else:
                skipped[name] = "Layer not found in target model"
        
        model.load_state_dict(model_dict, strict=False)
        
        # Update transfer summary
        if not transfer_summary.get('transfer_stats'):
            transfer_summary = {
                'transferred': transferred,
                'skipped': skipped,
                'transfer_stats': {
                    'total_attempted': len(state_dict),
                    'successfully_transferred': len(transferred),
                    'skipped_count': len(skipped)
                }
            }
        
        if logger:
            logger.info(f"Loaded PyTorch checkpoint: {len(transferred)} layers transferred, {len(skipped)} skipped")
            
    elif synthseg_path.suffix == '.h5':
        # Use H5 transfer method
        if logger:
            logger.info("Detected .h5 file - using H5 weight transfer")
        
        transfer_summary = transfer_synthseg_weights(
            h5_path=str(synthseg_path),
            torch_model=model,
            transfer_encoder=True,
            transfer_decoder=True,
            freeze_seg_layers=False  # We'll handle freezing separately
        )
    else:
        raise ValueError(f"Unsupported file format: {synthseg_path.suffix}. Expected .pth or .h5")
    
    # Apply freezing strategy
    frozen_layers = []
    for name, param in model.named_parameters():
        should_freeze = False
        
        if freeze_encoder and 'encoder' in name:
            should_freeze = True
        elif freeze_decoder and 'seg_head' in name:
            should_freeze = True
            
        if should_freeze:
            param.requires_grad = False
            frozen_layers.append(name)
    
    if frozen_layers and logger:
        logger.info(f"Frozen {len(frozen_layers)} layers based on freezing strategy")
        logger.info(f"Freeze encoder: {freeze_encoder}, Freeze decoder: {freeze_decoder}")
    
    return transfer_summary


def main() -> None:

    # 1. ─── configuration & reproducibility ─────────────────── #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/multitask/finetune.yaml"
    cfg = Config(cfg_file)
    
    # 2. ─── experiment naming / I/O ─────────────────────────── #
    experiment_name = cfg.get("output.experiment_name")
    if not experiment_name:
        experiment_name = f'multitask_synthseg_{timestamp}'
    out_root = Path(cfg.get("output.output_dir", "output"))
    ckpt_dir = out_root / "checkpoints" / experiment_name
    log_dir = out_root / "logs" / experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("brain-age-synthseg", log_file=log_dir / "train.log")
    
    logger.info("Initializing configuration...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Experiment: {experiment_name}\nConfig: {cfg_file}")
    logger.info(f"Using config: {cfg.config}")

    # 3. ─── Label configuration ─────────────────────────────── #
    # CRITICAL: SynthSeg has 33 classes (including background)
    n_classes = 33  # From SynthSeg labels table: 33 total classes
    logger.info(f"Using {n_classes} classes for segmentation (SynthSeg format)")
    logger.info(f"Generation labels: {len(GENERATION_LABELS)} unique labels")
    logger.info(f"Generation classes: {len(GENERATION_CLASSES)} mapped classes")
    
    # Verify we have the right number
    assert len(GENERATION_CLASSES) == n_classes, f"Mismatch: GENERATION_CLASSES has {len(GENERATION_CLASSES)} classes, expected {n_classes}"

    # 4. ─── W&B init ─────────────────────────────────────────── #
    logger.info("Initializing Weights & Biases...")
    use_wandb = cfg.get("wandb.use_wandb", True)
    if use_wandb:
        logger.info("Setting up W&B tracking")
        wandb.login(key="2abdb867a9244072f2237704a3cacc77fa548dd8")
        wandb.init(
            project=cfg.get("wandb.project", "brain-age-synthseg"),
            entity=cfg.get("wandb.entity"),
            name=experiment_name,
            config=cfg.config,
            reinit=True,
        )
        cfg.save_config(ckpt_dir / "config.yaml")

    # 5. ─── device ───────────────────────────────────────────── #
    logger.info("Setting up device...")
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 6. ─── Brain Generator ──────────────────────────────────── #
    logger.info("Initializing Brain Generator...")
    bg_cfg = cfg.get("brain_generator", {})
    
    # Prior distribution parameters - using 33 classes
    mean_loc = bg_cfg.get("mean_loc", 125.0)
    mean_scale = bg_cfg.get("mean_scale", 125.0)
    std_loc = bg_cfg.get("std_loc", 17.5)
    std_scale = bg_cfg.get("std_scale", 17.5)
    
    # Prior distribution parameters - using GENERATION_CLASSES (15 classes, not 33!)
    n_intensity_classes = len(np.unique(GENERATION_CLASSES))  # = 15
    n_output_classes = len(GENERATION_LABELS)                 # = 33

    prior_means = np.vstack([
        np.full(n_intensity_classes, mean_loc, dtype=float),    # 15 classes for intensity
        np.full(n_intensity_classes, mean_scale, dtype=float),
    ])

    prior_stds = np.vstack([
        np.full(n_intensity_classes, std_loc, dtype=float),     # 15 classes for intensity  
        np.full(n_intensity_classes, std_scale, dtype=float),
    ])


    # Set background class (label 0) to zero
    prior_means[:, 0] = 0.0    
    prior_stds[:, 0] = 0.0     
    
    brain_generator = BABrainGenerator(
        return_segmentation=True,
        prior_means=prior_means,
        prior_stds=prior_stds,
        distribution=bg_cfg.get("distribution", "normal"),
        prob=bg_cfg.get("prob", {}),
        rotation_range=bg_cfg.get("rotation_range", 10),
        scaling_range=bg_cfg.get("scaling_range", 0.1),
        shear_bounds=bg_cfg.get("shear_bounds", 0.005),
        translation_bounds=bg_cfg.get("translation_bounds", False),
        contrast_range=tuple(bg_cfg.get("contrast_range", [0.8, 1.2])),
        log_gamma_std=bg_cfg.get("log_gamma_std", 0.1),
        shift_offset=bg_cfg.get("shift_offset", 0.1),
        hist_control_points=bg_cfg.get("hist_control_points", 5),
        noise_mean=bg_cfg.get("noise_mean", 0.5),
        noise_std=bg_cfg.get("noise_std", 0.08),
        rician_std=bg_cfg.get("rician_std", 0.08),
        gibbs_alpha=bg_cfg.get("gibbs_alpha", 0.5),
        blur_sigma=bg_cfg.get("blur_sigma", 1.0),
        bias_field_rng=tuple(bg_cfg.get("bias_field_rng", [0.0, 0.8])),
        min_res=bg_cfg.get("min_res", 1.0),
        max_res_iso=bg_cfg.get("max_res_iso", 1.8),
        output_shape=tuple(bg_cfg.get("output_shape", [160, 192, 160])),
        # IMPORTANT: Keep original labels for loss computation!
        generation_labels=GENERATION_LABELS,
        output_labels=GENERATION_CLASSES,  # Don't convert - keep original labels!
    )
    
    validation_generator = ValidationGenerator(
        segmented_data_dir=Path(cfg.get("data.segmented_data_dir")),
        return_segmentation=True,
        use_intensity_clip_normalize=True,
        # IMPORTANT: Keep original labels for loss computation!
        generation_labels=GENERATION_LABELS,
        output_labels=GENERATION_LABELS,  # Don't convert - keep original labels!
    )

    # 7. ─── CSV → dataset / sampler ─────────────────────────── #
    logger.info("Reading CSV files...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv = Path(cfg.get("data.val_csv"))
    test_csv = Path(cfg.get("data.test_csv"))
    segmented_data_dir = Path(cfg.get("data.segmented_data_dir"))
    real_data_dir = Path(cfg.get("data.real_data_dir"))

    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w, train_s, train_m = read_csv(train_csv, segmented_data_dir)
    logger.info(f"Reading val CSV from {val_csv}")
    val_p, val_a, val_w, val_s, val_m = read_csv(val_csv, real_data_dir)
    logger.info(f"Reading test CSV from {test_csv}")
    test_p, test_a, test_w, test_s, test_m = read_csv(test_csv, real_data_dir)

    logger.info(f"Train={len(train_p)}  Val={len(val_p)}  Test={len(test_p)}")

    print("=== AGE RANGES ===")
    print(f"Train ages: min={min(train_a):.2f}, max={max(train_a):.2f}, mean={np.mean(train_a):.2f}, std={np.std(train_a):.2f}")
    print(f"Val ages:   min={min(val_a):.2f}, max={max(val_a):.2f}, mean={np.mean(val_a):.2f}, std={np.std(val_a):.2f}")
    print(f"Test ages:  min={min(test_a):.2f}, max={max(test_a):.2f}, mean={np.mean(test_a):.2f}, std={np.std(test_a):.2f}")

    logger.info("Initializing datasets...")
    train_ds = BADataset(
        file_paths=train_p, age_labels=train_a, sample_wts=train_w,
        sexes=train_s, modalities=train_m, transform=brain_generator, mode="train",
    )
    val_ds = BADataset(
        file_paths=val_p, age_labels=val_a, sample_wts=val_w,
        sexes=val_s, modalities=val_m, transform=validation_generator, mode="val",
    )
    test_ds = BADataset(
        file_paths=test_p, age_labels=test_a, sample_wts=test_w,
        sexes=test_s, modalities=test_m, transform=validation_generator, mode="test",
    )

    sampler = WeightedRandomSampler(weights=train_w, num_samples=len(train_w), replacement=True)
    dl_kwargs = dict(
        num_workers=cfg.get("data.num_workers", 6),
        pin_memory=cfg.get("data.pin_memory", True),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.get("training.batch_size", 8), sampler=sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("training.batch_size", 8), shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("training.batch_size", 8), shuffle=False, **dl_kwargs)

    # 8. ─── model ─────────────────────────────────────────────── #
    logger.info("Initializing MultiTaskBrainAge model...")
    model = MultiTaskBrainAge(n_classes=n_output_classes).to(device)
    logger.info(f"Model initialized with {n_classes} segmentation classes")
    print(f"Model hyperparameters: {cfg.get('model')}")

    # 9. ─── Load SynthSeg weights ────────────────────────────── #
    synthseg_cfg = cfg.get("synthseg", {})
    synthseg_path = synthseg_cfg.get("model_path")
    
    if synthseg_path and Path(synthseg_path).exists():
        logger.info("=" * 60)
        logger.info("LOADING SYNTHSEG WEIGHTS")
        logger.info("=" * 60)
        
        freeze_encoder = synthseg_cfg.get("freeze_encoder", False)
        freeze_decoder = synthseg_cfg.get("freeze_decoder", False)
        
        try:
            transfer_summary = load_synthseg_weights(
                model=model,
                synthseg_path=synthseg_path,
                freeze_encoder=freeze_encoder,
                freeze_decoder=freeze_decoder,
                logger=logger
            )
            
            # Log transfer results to wandb
            if use_wandb:
                wandb.log({
                    "synthseg_transfer_rate": transfer_summary['transfer_stats']['successfully_transferred'] / transfer_summary['transfer_stats']['total_attempted'],
                    "synthseg_transferred_layers": transfer_summary['transfer_stats']['successfully_transferred'],
                    "synthseg_total_layers": transfer_summary['transfer_stats']['total_attempted']
                })
            
            logger.info("SynthSeg weights loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load SynthSeg weights: {e}")
            logger.info("Continuing with random initialization...")
    else:
        logger.warning(f"SynthSeg model path not found or not specified: {synthseg_path}")
        logger.info("Training with random initialization...")

    # Check for additional checkpoint loading
    checkpoint_path = cfg.get("model.checkpoint")
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, device, logger)
    
    if use_wandb:
        wandb.watch(model, log="all", log_graph=False)

    # 10. ─── trainer ──────────────────────────────────────────── #
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
        wandb_project=cfg.get("wandb.project", "brain-age-synthseg"),
        wandb_entity=cfg.get("wandb.entity"),
        wandb_config=cfg.config,
        experiment_name=experiment_name,
    )
    logger.info("Trainer initialized")

    # 11. ─── train ───────────────────────────────────────────── #
    logger.info("Starting multi-task training with SynthSeg initialization...")
    try:
        t0 = time.time()
        results = trainer.train()
        history = results["history"]
        best_mae_info = results["best_mae_info"]
        
        logger.info(f"Training finished in {time.time()-t0:.1f}s")
        
        # Save results
        results_to_save = {
            "history": history,
            "best_mae_info": best_mae_info,
            "synthseg_config": synthseg_cfg,
            "n_classes": n_classes,
            "generation_classes_range": f"{GENERATION_CLASSES.min()}-{GENERATION_CLASSES.max()}",
        }
        
        if 'transfer_summary' in locals():
            results_to_save["synthseg_transfer_summary"] = transfer_summary
        
        json.dump(results_to_save, (ckpt_dir / "training_results.json").open("w"), indent=2)
        
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

    # 12. ─── evaluate ───────────────────────────────────────── #
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
