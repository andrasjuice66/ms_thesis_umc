#!/usr/bin/env python
"""
Standalone evaluation script for brain age prediction models.

Supports 3-fold evaluation regime:
1. Normal test evaluation
2. Domain randomized evaluation
3. Domain randomized + tumor simulation evaluation

Usage:
    python brain_age_pred/scripts/evaluate.py --model SFCN --model_path path/to/model.pt --config configs/evaluate/evaluate.yaml
"""

import os, sys, time, json, argparse
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
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.dom_rand.domain_randomization import DomainRandomizer
from brain_age_pred.models.sfcn import SFCN
from brain_age_pred.models.sfcn_class import SFCNClass
from brain_age_pred.models.brainagenext import BrainAgeNeXt
from brain_age_pred.training.trainer import BrainAgeTrainer
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv, load_checkpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate brain age prediction models")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["SFCN", "sfcn", "SFCN_Class", "sfcn_class", "BrainAgeNeXt", "brainagenext"],
                       help="Model type to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    return parser.parse_args()


def create_model(model_type: str, model_config: dict, device: torch.device, logger):
    """Create model based on type and configuration"""
    model_type = model_type.lower()
    
    if model_type == "sfcn":
        logger.info("Creating SFCN model")
        model = SFCN(
            in_channels=model_config.get("in_channels", 1),
            dropout_rate=model_config.get("dropout_rate", 0.5),
            age_min=model_config.get("age_min", 20),
            age_max=model_config.get("age_max", 80),
        ).to(device)
    elif model_type in ["sfcn_class", "sfcnclass"]:
        logger.info("Creating SFCN Class model")
        model = SFCNClass(
            in_channels=model_config.get("in_channels", 1),
            dropout_rate=model_config.get("dropout_rate", 0.5),
            channels=model_config.get("channels", (32, 64, 128, 256, 256, 64)),
            age_min=model_config.get("age_min", 20),
            age_max=model_config.get("age_max", 80),
        ).to(device)
    elif model_type == "brainagenext":
        logger.info("Creating BrainAgeNext model")
        model = BrainAgeNeXt(
            in_channels=model_config.get("in_channels", 1),
            dropout_rate=model_config.get("dropout_rate", 0.5),
            model_id=model_config.get("model_id", "B"),
            kernel_size=model_config.get("kernel_size", 3),
            deep_supervision=model_config.get("deep_supervision", True),
            feature_size=model_config.get("feature_size", 512),
            hidden_size=model_config.get("hidden_size", 64),
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_eval_transforms(cfg: Config, device: torch.device, logger, use_domain_rand=False, use_tumor=False):
    """Create evaluation-specific transforms"""
    if not use_domain_rand:
        return None
    
    # Create domain randomization for evaluation using same config as training
    eval_rand_cfg = cfg.get("domain_randomization", {}).copy()
    eval_tumor_cfg = eval_rand_cfg.get("tumor_config", {}).copy() if use_tumor else {}
    
    # IMPORTANT: For tumor evaluation, always set probability to 1.0 to ensure tumors are always added
    if use_tumor and eval_tumor_cfg:
        original_prob = eval_tumor_cfg.get("prob", cfg.get("domain_randomization", {}).get("tumor_config", {}).get("prob", "unknown"))
        eval_tumor_cfg["prob"] = 1.0
        logger.info(f"Overriding tumor probability to 1.0 for evaluation (was {original_prob})")
    
    # Remove conflicting keys that we want to override to prevent config conflicts
    eval_rand_cfg.pop("use_domain_randomization", None)
    eval_rand_cfg.pop("use_tumor_simulation", None)
    eval_rand_cfg.pop("tumor_config", None)
    
    # Explicitly log what regime we're creating
    if use_tumor:
        logger.info("Creating domain randomization + tumor simulation transforms for evaluation")
    else:
        logger.info("Creating domain randomization (no tumor) transforms for evaluation")
    
    eval_transform = DomainRandomizer(
        device=device,
        use_domain_randomization=True,
        use_tumor_simulation=use_tumor,
        tumor_config=eval_tumor_cfg if use_tumor else None,
        **eval_rand_cfg,
    )
    
    return eval_transform


def create_evaluation_tables(normal_metrics, dom_rand_metrics, dom_rand_tumor_metrics):
    """Create wandb tables summarizing evaluation metrics by modality"""
    
    # Get unique modalities from the metrics keys
    modalities = set()
    for metrics in [normal_metrics, dom_rand_metrics, dom_rand_tumor_metrics]:
        for key in metrics.keys():
            if '_mae' in key and key != 'mae':
                modality = key.replace('_mae', '')
                if modality not in ['mae_std']:  # Skip std metrics
                    modalities.add(modality)
    
    modalities = sorted(list(modalities))
    
    # Create table data
    table_data = []
    
    # Add overall (average) row
    table_data.append([
        "Average",
        f"{normal_metrics['mae']:.4f}",
        f"{normal_metrics['mse']:.4f}",
        f"{normal_metrics['r2']:.4f}",
        f"{normal_metrics['correlation']:.4f}",
        f"{dom_rand_metrics['mae']:.4f} ± {dom_rand_metrics.get('mae_std', 0):.4f}",
        f"{dom_rand_metrics['mse']:.4f} ± {dom_rand_metrics.get('mse_std', 0):.4f}",
        f"{dom_rand_metrics['r2']:.4f} ± {dom_rand_metrics.get('r2_std', 0):.4f}",
        f"{dom_rand_metrics['correlation']:.4f} ± {dom_rand_metrics.get('correlation_std', 0):.4f}",
        f"{dom_rand_tumor_metrics['mae']:.4f} ± {dom_rand_tumor_metrics.get('mae_std', 0):.4f}",
        f"{dom_rand_tumor_metrics['mse']:.4f} ± {dom_rand_tumor_metrics.get('mse_std', 0):.4f}",
        f"{dom_rand_tumor_metrics['r2']:.4f} ± {dom_rand_tumor_metrics.get('r2_std', 0):.4f}",
        f"{dom_rand_tumor_metrics['correlation']:.4f} ± {dom_rand_tumor_metrics.get('correlation_std', 0):.4f}",
    ])
    
    # Add modality-specific rows
    for modality in modalities:
        # Get metrics for this modality (with fallbacks)
        normal_mae = normal_metrics.get(f"{modality}_mae", 0)
        normal_mse = normal_metrics.get(f"{modality}_mse", 0)
        normal_r2 = normal_metrics.get(f"{modality}_r2", 0)
        normal_corr = normal_metrics.get(f"{modality}_correlation", 0)
        
        dom_rand_mae = dom_rand_metrics.get(f"{modality}_mae", 0)
        dom_rand_mse = dom_rand_metrics.get(f"{modality}_mse", 0)
        dom_rand_r2 = dom_rand_metrics.get(f"{modality}_r2", 0)
        dom_rand_corr = dom_rand_metrics.get(f"{modality}_correlation", 0)
        dom_rand_mae_std = dom_rand_metrics.get(f"{modality}_mae_std", 0)
        dom_rand_mse_std = dom_rand_metrics.get(f"{modality}_mse_std", 0)
        dom_rand_r2_std = dom_rand_metrics.get(f"{modality}_r2_std", 0)
        dom_rand_corr_std = dom_rand_metrics.get(f"{modality}_correlation_std", 0)
        
        tumor_mae = dom_rand_tumor_metrics.get(f"{modality}_mae", 0)
        tumor_mse = dom_rand_tumor_metrics.get(f"{modality}_mse", 0)
        tumor_r2 = dom_rand_tumor_metrics.get(f"{modality}_r2", 0)
        tumor_corr = dom_rand_tumor_metrics.get(f"{modality}_correlation", 0)
        tumor_mae_std = dom_rand_tumor_metrics.get(f"{modality}_mae_std", 0)
        tumor_mse_std = dom_rand_tumor_metrics.get(f"{modality}_mse_std", 0)
        tumor_r2_std = dom_rand_tumor_metrics.get(f"{modality}_r2_std", 0)
        tumor_corr_std = dom_rand_tumor_metrics.get(f"{modality}_correlation_std", 0)
        
        table_data.append([
            modality,
            f"{normal_mae:.4f}",
            f"{normal_mse:.4f}",
            f"{normal_r2:.4f}",
            f"{normal_corr:.4f}",
            f"{dom_rand_mae:.4f} ± {dom_rand_mae_std:.4f}",
            f"{dom_rand_mse:.4f} ± {dom_rand_mse_std:.4f}",
            f"{dom_rand_r2:.4f} ± {dom_rand_r2_std:.4f}",
            f"{dom_rand_corr:.4f} ± {dom_rand_corr_std:.4f}",
            f"{tumor_mae:.4f} ± {tumor_mae_std:.4f}",
            f"{tumor_mse:.4f} ± {tumor_mse_std:.4f}",
            f"{tumor_r2:.4f} ± {tumor_r2_std:.4f}",
            f"{tumor_corr:.4f} ± {tumor_corr_std:.4f}",
        ])
    
    # Create wandb table
    table = wandb.Table(
        columns=[
            "Modality",
            "Normal MAE", "Normal MSE", "Normal R²", "Normal Correlation",
            "Dom Rand MAE", "Dom Rand MSE", "Dom Rand R²", "Dom Rand Correlation", 
            "Dom Rand + Tumor MAE", "Dom Rand + Tumor MSE", "Dom Rand + Tumor R²", "Dom Rand + Tumor Correlation"
        ],
        data=table_data
    )
    
    return table


def run_multi_fold_evaluation(cfg: Config, model_type: str, model_path: str, device: torch.device, logger, 
                             test_p, test_a, test_s, test_m, transform, log_dir, n_folds=5, eval_name="test"):
    """Run evaluation multiple times with different augmentations and average results"""
    logger.info(f"Running {n_folds}-fold {eval_name} evaluation...")
    
    all_metrics = []
    
    # Setup data loader kwargs
    dl_kwargs = dict(
        num_workers=cfg.get("data.num_workers", 6),
        pin_memory=cfg.get("data.pin_memory", True),
        pin_memory_device=cfg.get("data.pin_memory_device"),
        persistent_workers=cfg.get("data.persistent_workers", True),
        prefetch_factor=cfg.get("data.prefetch_factor"),
    )
    
    for fold in range(n_folds):
        logger.info(f"{eval_name} evaluation fold {fold+1}/{n_folds}")
        
        # Create test dataset with transform
        eval_test_ds = BADataset(
            file_paths=test_p,
            age_labels=test_a,
            sexes=test_s,
            modalities=test_m,
            transform=transform,
            mode="test",
            cache_size=0,  # No caching for evaluation
        )
        
        # Create data loader
        eval_test_loader = torch.utils.data.DataLoader(
            eval_test_ds,
            batch_size=cfg.get("batch_size", 8),
            shuffle=False,
            **dl_kwargs,
        )
        
        # Create model and load checkpoint for this fold
        model = create_model(model_type, cfg.get("model", {}), device, logger)
        
        # Load checkpoint
        try:
            checkpoint_info = load_checkpoint(model, model_path, device, logger)
            if checkpoint_info:
                logger.info(f"Loaded checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Create trainer for evaluation
        trainer = BrainAgeTrainer(
            model=model,
            train_loader=None,  # Not needed for evaluation
            val_loader=None,    # Not needed for evaluation
            test_loader=eval_test_loader,
            config={},  # Minimal config for evaluation
            device=device,
            checkpoint_dir="/home/ajoos/brain_age_pred/output/checkpoints/",
            log_dir=log_dir,
            use_wandb=False,  # Disable wandb in trainer, we handle it here
            age_min=cfg.get("data.age_min", 20),
            age_max=cfg.get("data.age_max", 80),
        )
        
        # Run evaluation
        metrics = trainer.evaluate(eval_test_loader, checkpoint_path=model_path)
        all_metrics.append(metrics)
    
    # Average metrics across folds
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)
    
    logger.info(f"{eval_name} evaluation results (averaged over {n_folds} folds):")
    logger.info(f"MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['mae_std']:.4f}")
    logger.info(f"MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['mse_std']:.4f}")
    logger.info(f"R²: {avg_metrics['r2']:.4f} ± {avg_metrics['r2_std']:.4f}")
    
    return avg_metrics


def main() -> None:
    # Parse command line arguments
    args = parse_args()
    
    # 1. ─── configuration & reproducibility ─────────────────── #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = Config(args.config)
    
    # 2. ─── experiment naming / I/O ─────────────────────────── #
    model_name = args.model.lower()
    checkpoint_name = Path(args.model_path).stem
    experiment_name = f"evaluation_{model_name}_{checkpoint_name}_{timestamp}"
    
    out_root = Path(cfg.get("output_dir", "output/evaluations"))
    log_dir = out_root / "logs" / experiment_name
    results_dir = out_root / "results" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("brain-age-eval", log_file=log_dir / "evaluate.log")
    logger.info(f"Evaluation experiment: {experiment_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Model path: {args.model_path}")
    
    set_seed(cfg.get("seed", 42))
    
    # Validate model path
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    # 3. ─── W&B init ─────────────────────────────────────────── #
    wandb_config = cfg.get("wandb", {})
    use_wandb = bool(wandb_config)
    
    if use_wandb:
        logger.info("Setting up W&B tracking")
        wandb_api_key = wandb_config.get("api_key")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        # Add command line args to config for logging
        wandb_cfg = cfg.config.copy()
        wandb_cfg.update({
            "cli_model": args.model,
            "cli_model_path": args.model_path,
            "cli_config": args.config
        })
        
        wandb.init(
            project=wandb_config.get("project", "brain-age-evaluation"),
            entity=wandb_config.get("entity"),
            name=experiment_name,
            config=wandb_cfg,
            reinit=True,
        )
    
    # 4. ─── device ───────────────────────────────────────────── #
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")
    
    # 5. ─── load datasets ────────────────────────────────────── #
    data_dir = Path(cfg.get("data_dir", "."))
    
    # Determine which datasets to evaluate
    datasets_to_eval = cfg.get("datasets", ["test"])
    dataset_paths = cfg.get("dataset_paths", {})
    
    # For now, let's focus on test dataset (can be extended)
    if "test" in datasets_to_eval:
        test_csv = Path(dataset_paths.get("test"))
        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")
        
        logger.info(f"Reading test CSV from {test_csv}")
        test_p, test_a, test_w, test_s, test_m = read_csv(test_csv, data_dir)
        logger.info(f"Test set: {len(test_p)} samples")
        
        # Log age statistics
        logger.info(f"Test ages: min={min(test_a):.2f}, max={max(test_a):.2f}, mean={np.mean(test_a):.2f}")
    
    # 6. ─── evaluation setup ─────────────────────────────────── #
    eval_config = cfg.get("evaluation", {})
    
    # Setup data loader kwargs
    dl_kwargs = dict(
        num_workers=cfg.get("data.num_workers", 6),
        pin_memory=cfg.get("data.pin_memory", False),
        pin_memory_device=cfg.get("data.pin_memory_device"),
        persistent_workers=cfg.get("data.persistent_workers", True),
        prefetch_factor=cfg.get("data.prefetch_factor"),
    )
    
    # Create normal test dataset (no transforms)
    normal_test_ds = BADataset(
        file_paths=test_p,
        age_labels=test_a,
        sexes=test_s,
        modalities=test_m,
        transform=None,
        mode="test",
        cache_size=0,
    )
    
    normal_test_loader = torch.utils.data.DataLoader(
        normal_test_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        **dl_kwargs,
    )
    
    # 7. ─── run evaluations ──────────────────────────────────── #
    results = {}
    
    try:
        # 1. Normal test evaluation
        if eval_config.get("normal_test", True):
            logger.info("=== 1/3: Normal test evaluation ===")
            
            # Create model and load checkpoint
            model = create_model(args.model, cfg.get("model", {}), device, logger)
            checkpoint_info = load_checkpoint(model, args.model_path, device, logger)
            if checkpoint_info:
                logger.info(f"Loaded checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
            
            # Create trainer for evaluation
            trainer = BrainAgeTrainer(
                model=model,
                train_loader=None,
                val_loader=None,
                test_loader=normal_test_loader,
                config={},
                device=device,
                checkpoint_dir="/home/ajoos/brain_age_pred/output/checkpoints/",
                log_dir=log_dir,
                use_wandb=False,
                age_min=cfg.get("data.age_min", 18),
                age_max=cfg.get("data.age_max", 90),
            )
            
            normal_metrics = trainer.evaluate(normal_test_loader, checkpoint_path=args.model_path)
            results["normal"] = normal_metrics
            logger.info(f"Normal test results: MAE={normal_metrics['mae']:.4f}")
        
        # 2. Domain randomized test evaluation
        dom_rand_config = eval_config.get("domain_randomized", {})
        if dom_rand_config.get("enabled", True):
            logger.info("=== 2/3: Domain randomized test evaluation ===")
            dom_rand_transform = create_eval_transforms(cfg, device, logger, use_domain_rand=True, use_tumor=False)
            
            n_folds = dom_rand_config.get("n_folds", 5)
            dom_rand_metrics = run_multi_fold_evaluation(
                cfg, args.model, args.model_path, device, logger, test_p, test_a, test_s, test_m,
                dom_rand_transform, log_dir, n_folds=n_folds, eval_name="domain_randomized"
            )
            results["domain_randomized"] = dom_rand_metrics
        
        # 3. Domain randomized + tumor simulation test evaluation
        dom_rand_tumor_config = eval_config.get("domain_rand_tumor", {})
        if dom_rand_tumor_config.get("enabled", True):
            logger.info("=== 3/3: Domain randomized + tumor simulation test evaluation ===")
            dom_rand_tumor_transform = create_eval_transforms(cfg, device, logger, use_domain_rand=True, use_tumor=True)
            
            n_folds = dom_rand_tumor_config.get("n_folds", 5)
            dom_rand_tumor_metrics = run_multi_fold_evaluation(
                cfg, args.model, args.model_path, device, logger, test_p, test_a, test_s, test_m,
                dom_rand_tumor_transform, log_dir, n_folds=n_folds, eval_name="domain_rand_tumor"
            )
            results["domain_rand_tumor"] = dom_rand_tumor_metrics
        
        # 8. ─── log results ───────────────────────────────────── #
        if use_wandb:
            # Log results to W&B
            if "normal" in results:
                wandb.log({f"test/{k}": v for k, v in results["normal"].items()})
            
            if "domain_randomized" in results:
                wandb.log({f"test_dom_rand/{k}": v for k, v in results["domain_randomized"].items()})
            
            if "domain_rand_tumor" in results:
                wandb.log({f"test_dom_rand_tumor/{k}": v for k, v in results["domain_rand_tumor"].items()})
            
            # Create comparison summary
            if len(results) == 3:
                wandb.log({
                    "evaluation_summary/normal_mae": results["normal"]["mae"],
                    "evaluation_summary/dom_rand_mae": results["domain_randomized"]["mae"],
                    "evaluation_summary/dom_rand_tumor_mae": results["domain_rand_tumor"]["mae"],
                    "evaluation_summary/dom_rand_mae_std": results["domain_randomized"]["mae_std"],
                    "evaluation_summary/dom_rand_tumor_mae_std": results["domain_rand_tumor"]["mae_std"],
                })
                
                # Create and log evaluation table
                logger.info("Creating evaluation summary table for W&B...")
                evaluation_table = create_evaluation_tables(
                    results["normal"], 
                    results["domain_randomized"], 
                    results["domain_rand_tumor"]
                )
                wandb.log({"test_evaluation_summary": evaluation_table})
        
        # Save results to JSON
        json.dump(results, open(results_dir / "evaluation_results.json", "w"), indent=2)
        
        # 9. ─── summary ─────────────────────────────────────── #
        logger.info("=== Evaluation Summary ===")
        if "normal" in results:
            logger.info(f"Normal test MAE: {results['normal']['mae']:.4f}")
        if "domain_randomized" in results:
            dr_mae = results['domain_randomized']['mae']
            dr_std = results['domain_randomized']['mae_std']
            logger.info(f"Domain rand test MAE: {dr_mae:.4f} ± {dr_std:.4f}")
        if "domain_rand_tumor" in results:
            drt_mae = results['domain_rand_tumor']['mae']
            drt_std = results['domain_rand_tumor']['mae_std']
            logger.info(f"Domain rand + tumor test MAE: {drt_mae:.4f} ± {drt_std:.4f}")
        
        logger.info(f"Results saved to: {results_dir / 'evaluation_results.json'}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if use_wandb:
            wandb.finish()
        logger.info("Evaluation complete.")


if __name__ == "__main__":
    main() 
    