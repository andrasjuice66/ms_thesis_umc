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
from brain_age_pred.dataset.domain_randomization import DomainRandomizer
from brain_age_pred.models.sfcn import SFCN
from brain_age_pred.models.sfcn_class import SFCNClass
from brain_age_pred.models.brainagenext import BrainAgeNeXt
from brain_age_pred.training.trainer import BrainAgeTrainer
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv, load_checkpoint
from torch.utils.data import WeightedRandomSampler, DataLoader



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

    # 5. ─── transforms (GPU-ready) with tumor simulation ─────── #
    logger.info("Initializing domain randomization transforms...")
    rand_cfg = cfg.get("domain_randomization", {})
    if rand_cfg.get("use_domain_randomization", False):
        transform = DomainRandomizer(
            **rand_cfg,
        )
        
        if rand_cfg.get("use_tumor_simulation", False):
            tumor_cfg = rand_cfg.get("tumor_config", {})
            logger.info(f"✓ Tumor simulation enabled with probability: {tumor_cfg.get('prob', 0.3)}")
            
            if tumor_cfg.get("use_age_based_segmentation", False):
                logger.info("✓ Using age-based segmentation for tumor placement:")
                seg_paths = tumor_cfg.get("segmentation_paths", {})
                age_ranges = tumor_cfg.get("age_ranges", {})
                
                # Validate segmentation files exist
                missing_files = []
                for age_group, seg_path in seg_paths.items():
                    if not Path(seg_path).exists():
                        missing_files.append(f"{age_group}: {seg_path}")
                    else:
                        age_range = age_ranges.get(age_group, "unknown")
                        logger.info(f"  • {age_group}: {seg_path} (ages {age_range})")
                
                if missing_files:
                    logger.error("Missing segmentation files:")
                    for missing in missing_files:
                        logger.error(f"  ✗ {missing}")
                    raise FileNotFoundError("Required segmentation files not found")
                
                logger.info("All segmentation files found and will be preloaded")
            else:
                logger.info("Using intensity-based brain mask for tumor placement")
                
            # Log tumor generation parameters
            logger.info(f"Tumor parameters: perlin_res={tumor_cfg.get('perlin_res', [2,2,2])}, "
                       f"size_range={tumor_cfg.get('tumor_size_factor_range', [0.5, 2.0])}, "
                       f"fluid_dynamics={tumor_cfg.get('use_fluid_dynamics', True)}")
        else:
            logger.info("Tumor simulation disabled")
    else:
        transform = None
        logger.info("Domain randomization disabled")
    
    logger.info(f"Domain randomizer initialized: {rand_cfg.get('use_domain_randomization', False)}")

    # 6. ─── CSV → dataset / sampler ─────────────────────────── #
    logger.info("Reading CSV files...")
    train_csv = Path(cfg.get("data.train_csv"))
    val_csv   = Path(cfg.get("data.val_csv"))
    test_csv  = Path(cfg.get("data.test_csv"))
    real_data_dir = Path(cfg.get("data.real_data_dir"))
    


    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w, train_s, train_m = read_csv(
        train_csv,
        real_data_dir,
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
    print("=== AGE RANGES DEBUG ===")
    print(f"Train ages: min={min(train_a):.2f}, max={max(train_a):.2f}, mean={np.mean(train_a):.2f}")
    print(f"Val ages:   min={min(val_a):.2f}, max={max(val_a):.2f}, mean={np.mean(val_a):.2f}")
    print(f"Test ages:  min={min(test_a):.2f}, max={max(test_a):.2f}, mean={np.mean(test_a):.2f}")
    print("Sample train ages:", train_a[:5])
    print("Sample val ages:", val_a[:5])

    logger.info("Initializing datasets...")
    logger.info("Creating training dataset")

    train_ds = BADataset(
        file_paths   = train_p,
        age_labels   = train_a,
        sample_wts   = train_w,
        sexes        = train_s,
        modalities   = train_m,
        transform    = transform,
        mode         = "train",
        cache_size   = cfg.get("data.cache_size", 0),
    )
    
    logger.info("Creating validation dataset")
    val_ds   = BADataset(
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

    # 10. ─── 3-fold evaluation ─────────────────────────────────────── #
    def create_eval_transforms(use_domain_rand=False, use_tumor=False):
        """Create evaluation-specific transforms"""
        if not use_domain_rand:
            return None
        
        # Create domain randomization for evaluation using same config as training
        eval_rand_cfg = cfg.get("domain_randomization", {}).copy()
        eval_tumor_cfg = eval_rand_cfg.get("tumor_config", {}).copy() if use_tumor else {}
        
        # IMPORTANT: For tumor evaluation, always set probability to 1.0 to ensure tumors are always added
        if use_tumor and eval_tumor_cfg:
            eval_tumor_cfg["prob"] = 1.0
            logger.info(f"Overriding tumor probability to 1.0 for evaluation (was {cfg.get('domain_randomization', {}).get('tumor_config', {}).get('prob', 'unknown')})")
        
        # Remove conflicting keys that we want to override
        eval_rand_cfg.pop("use_domain_randomization", None)
        eval_rand_cfg.pop("use_tumor_simulation", None)
        eval_rand_cfg.pop("tumor_config", None)
        
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

    def run_multi_fold_evaluation(transform, n_folds=10, eval_name="test"):
        """Run evaluation multiple times with different augmentations and average results"""
        logger.info(f"Running {n_folds}-fold {eval_name} evaluation...")
        
        all_metrics = []
        
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
                batch_size=cfg.get("training.batch_size", 8),
                shuffle=False,
                **dl_kwargs,
            )
            
            # Run evaluation
            metrics = trainer.evaluate(eval_test_loader, checkpoint_path=best_mae_checkpoint)
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

    try:
        logger.info("Starting 3-fold evaluation using best MAE checkpoint...")
        best_mae_checkpoint = best_mae_info["checkpoint_path"]
        logger.info(f"Loading best checkpoint from epoch {best_mae_info['epoch']+1} with MAE {best_mae_info['value']:.4f}")
        
        # 1. Normal test evaluation
        logger.info("=== 1/3: Normal test evaluation ===")
        normal_metrics = trainer.evaluate(test_loader, checkpoint_path=best_mae_checkpoint)
        logger.info(f"Normal test results: {normal_metrics}")
        
        # 2. Domain randomized test evaluation (10 folds)
        logger.info("=== 2/3: Domain randomized test evaluation ===")
        dom_rand_transform = create_eval_transforms(use_domain_rand=True, use_tumor=False)
        dom_rand_metrics = run_multi_fold_evaluation(
            dom_rand_transform, n_folds=5, eval_name="domain_randomized"
        )
        
        # 3. Domain randomized + tumor simulation test evaluation (10 folds)
        logger.info("=== 3/3: Domain randomized + tumor simulation test evaluation ===")
        dom_rand_tumor_transform = create_eval_transforms(use_domain_rand=True, use_tumor=True)
        dom_rand_tumor_metrics = run_multi_fold_evaluation(
            dom_rand_tumor_transform, n_folds=5, eval_name="domain_rand_tumor"
        )
        
        # Log all results to W&B with appropriate prefixes
        if use_wandb:
            # Normal test results
            wandb.log({f"test/{k}": v for k, v in normal_metrics.items()})
            
            # Domain randomized results
            wandb.log({f"test_dom_rand/{k}": v for k, v in dom_rand_metrics.items()})
            
            # Domain randomized + tumor results
            wandb.log({f"test_dom_rand_tumor/{k}": v for k, v in dom_rand_tumor_metrics.items()})
            
            # Log summary comparison
            wandb.log({
                "evaluation_summary/normal_mae": normal_metrics["mae"],
                "evaluation_summary/dom_rand_mae": dom_rand_metrics["mae"],
                "evaluation_summary/dom_rand_tumor_mae": dom_rand_tumor_metrics["mae"],
                "evaluation_summary/dom_rand_mae_std": dom_rand_metrics["mae_std"],
                "evaluation_summary/dom_rand_tumor_mae_std": dom_rand_tumor_metrics["mae_std"],
            })
            
            # Create and log wandb table
            logger.info("Creating evaluation summary table for W&B...")
            evaluation_table = create_evaluation_tables(normal_metrics, dom_rand_metrics, dom_rand_tumor_metrics)
            wandb.log({"test_evaluation_summary": evaluation_table})
        
        # Save evaluation results
        eval_results = {
            "normal": normal_metrics,
            "domain_randomized": dom_rand_metrics,
            "domain_rand_tumor": dom_rand_tumor_metrics,
        }
        json.dump(eval_results, open(ckpt_dir/"evaluation_results.json","w"), indent=2)
        
        logger.info("=== Evaluation Summary ===")
        logger.info(f"Normal test MAE: {normal_metrics['mae']:.4f}")
        logger.info(f"Domain rand test MAE: {dom_rand_metrics['mae']:.4f} ± {dom_rand_metrics['mae_std']:.4f}")
        logger.info(f"Domain rand + tumor test MAE: {dom_rand_tumor_metrics['mae']:.4f} ± {dom_rand_tumor_metrics['mae_std']:.4f}")
        
    except Exception as e:
        logger.error(f"3-fold evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if use_wandb: wandb.finish()
        logger.info("All done.")
        return float(best_val)

if __name__ == "__main__":
    sys.exit(main())