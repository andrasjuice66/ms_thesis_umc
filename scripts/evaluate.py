#!/usr/bin/env python
"""
Simple script for evaluating a trained model from checkpoint.
"""
import os, sys
from pathlib import Path
from typing import Dict, Any

import torch
import wandb
from dotenv import load_dotenv

# ───────────────────── project imports ────────────────────── #
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dom_rand.dataset import BADataset
from brain_age_pred.models.sfcn import SFCN
from brain_age_pred.models.resnet3d import ResNet3D
from brain_age_pred.models.efficientnet3d import EfficientNet3D
from brain_age_pred.training.trainer import BrainAgeTrainer
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv, load_checkpoint

load_dotenv()

def evaluate_model(cfg_file: str) -> Dict[str, Any]:
    """
    Evaluate a model using configuration from file.
    
    Args:
        cfg_file: Path to config file
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # 1. ─── configuration ─────────────────────────────────── #
    cfg = Config(cfg_file)
    experiment_name = cfg.get("output.experiment_name", "evaluation")
    
    # Setup logging
    out_root = Path(cfg.get("output.output_dir", "output"))
    log_dir = out_root / "logs" / f"{experiment_name}_eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("brain-age-eval", log_file=log_dir / "eval.log")
    
    logger.info("Initializing configuration...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Config: {cfg_file}")

    # 2. ─── W&B init ─────────────────────────────────────── #
    if cfg.get("wandb.use_wandb", True):
        logger.info("Initializing Weights & Biases...")
        wandb.login(key=os.environ["WANDB_API"])
        wandb.init(
            project=cfg.get("wandb.project", "brain-age-pred"),
            entity=cfg.get("wandb.entity"),
            name=f"{experiment_name}_eval",
            config=cfg.config,
            reinit=True,
        )

    # 3. ─── device ───────────────────────────────────────── #
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 4. ─── dataset setup ───────────────────────────────── #
    logger.info("Reading test CSV...")
    test_csv = Path(cfg.get("data.test_csv"))
    data_dir = Path(cfg.get("data.data_dir"))
    
    test_p, test_a, _ = read_csv(test_csv, data_dir)
    
    test_ds = BADataset(
        file_paths=test_p,
        age_labels=test_a,
        transform=None,  # No domain randomization for evaluation
        mode="test",
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.get("training.batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("data.num_workers", 8),
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.get("data.persistent_workers", True),
        prefetch_factor=cfg.get("data.prefetch_factor", 4),
    )

    # 5. ─── model ───────────────────────────────────────── #
    logger.info("Initializing model...")
    mtype = cfg.get("model.type", "sfcn").lower()
    model_map = {"sfcn": SFCN, "resnet3d": ResNet3D, "efficientnet3d": EfficientNet3D}

    if mtype == "sfcn":
        model = SFCN(
            in_channels=cfg.get("model.in_channels"),
            dropout_rate=cfg.get("model.dropout_rate"),
        ).to(device)
    elif mtype == "resnet3d":
        model = ResNet3D(
            in_channels=cfg.get("model.in_channels"),
            dropout_rate=cfg.get("model.dropout_rate"),
            use_attention=cfg.get("model.use_attention", False),
        ).to(device)
    else:
        model = model_map[mtype](**cfg.get("model")).to(device)

    # Load checkpoint
    checkpoint_path = cfg.get("model.checkpoint")
    if not checkpoint_path:
        raise ValueError("No checkpoint path specified in config file")
        
    checkpoint_info = load_checkpoint(model, checkpoint_path, device, logger)
    if checkpoint_info:
        logger.info(f"Loaded checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
        if checkpoint_info.get('best_metric'):
            logger.info(f"Best metric from checkpoint: {checkpoint_info['best_metric']}")

    # 6. ─── evaluation ──────────────────────────────────── #
    logger.info("Initializing trainer for evaluation...")
    trainer = BrainAgeTrainer(
        model=model,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        config=cfg.get("training"),
        device=device,
        checkpoint_dir="output/checkpoints",
        log_dir=log_dir,
        use_wandb=cfg.get("wandb.use_wandb", True),
        wandb_project=cfg.get("wandb.project", "brain-age-pred"),
        wandb_entity=cfg.get("wandb.entity"),
        wandb_config=cfg.config,
        experiment_name=experiment_name,
    )

    try:
        logger.info("Running evaluation...")
        metrics = trainer.evaluate(test_loader)
        logger.info(f"Evaluation results: {metrics}")
        
        if cfg.get("wandb.use_wandb", True):
            wandb.log({"test/metrics": metrics})
            wandb.finish()
            
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        if cfg.get("wandb.use_wandb", True):
            wandb.finish()

def main():
    """Command line interface for evaluation."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <config_file>")
        sys.exit(1)
        
    cfg_file = sys.argv[1]
    
    try:
        metrics = evaluate_model(cfg_file)
        print("\nEvaluation Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
