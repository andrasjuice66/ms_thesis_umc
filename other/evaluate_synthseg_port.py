#!/usr/bin/env python
"""
Test script to load the ported PyTorch SynthSeg model and evaluate its segmentation performance.
This script validates that the PyTorch architecture correctly loads Keras weights and produces
segmentations comparable to the original SynthSeg model.
"""
import os, sys, time, json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dataset.dataset import BADataset
from brain_age_pred.models.synthseg_pytorch import SynthSeg
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv
from brain_age_pred.utils.weight_transfer import transfer_synthseg_weights
from brain_age_pred.brain_gen.validation_generator import ValidationGenerator
from brain_age_pred.brain_gen.labels import GENERATION_LABELS

def load_synthseg_weights(model: torch.nn.Module, 
                         synthseg_path: str, 
                         logger=None) -> dict:
    """
    Load SynthSeg weights into the PyTorch SynthSeg model.
    This function supports loading from both the original Keras .h5 file and a converted .pth checkpoint.
    """
    synthseg_path = Path(synthseg_path)
    if not synthseg_path.exists():
        raise FileNotFoundError(f"SynthSeg model not found: {synthseg_path}")
    
    if logger:
        logger.info(f"Loading SynthSeg weights from: {synthseg_path}")
    
    if synthseg_path.suffix == '.h5':
        if logger:
            logger.info("Detected .h5 file - using H5 weight transfer.")
        transfer_summary = transfer_synthseg_weights(
            h5_path=str(synthseg_path),
            torch_model=model,
            transfer_encoder=True,
            transfer_decoder=True,
            freeze_seg_layers=False
        )
    elif synthseg_path.suffix == '.pth':
        if logger:
            logger.info("Detected .pth file - loading as PyTorch state_dict.")
        checkpoint = torch.load(synthseg_path, map_location='cpu')
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        
        transfer_summary = checkpoint.get('transfer_summary', {
            'transfer_stats': {'successfully_transferred': len(state_dict), 'total_attempted': len(state_dict)}
        })
    else:
        raise ValueError(f"Unsupported file format: {synthseg_path.suffix}. Expected .pth or .h5")
        
    return transfer_summary


def compute_segmentation_metrics(predictions: torch.Tensor, 
                                targets: torch.Tensor, 
                                class_labels: np.ndarray) -> Dict[str, float]:
    """
    Computes Dice scores for each class.
    """
    if predictions.dim() == 5:  # Logits (B, C, D, H, W)
        preds = torch.argmax(predictions, dim=1)
    else: # Already argmax'd (B, D, H, W)
        preds = predictions
    
    preds_flat = preds.cpu().numpy().flatten()
    targets_flat = targets.cpu().numpy().flatten()
    
    metrics = {'accuracy': np.mean(preds_flat == targets_flat)}
    num_classes = len(class_labels)
    dice_scores = []
    
    for i in range(num_classes):
        pred_mask = (preds_flat == i)
        target_mask = (targets_flat == i)
        
        intersection = np.sum(pred_mask & target_mask)
        pred_sum = np.sum(pred_mask)
        target_sum = np.sum(target_mask)
        
        dice = (2.0 * intersection) / (pred_sum + target_sum) if (pred_sum + target_sum) > 0 else 1.0
        
        class_name = f"class_{class_labels[i]}"
        metrics[f'dice_{class_name}'] = dice
        dice_scores.append(dice)
    
    # Mean Dice (excluding background class 0)
    metrics['mean_dice'] = np.mean(dice_scores[1:]) if num_classes > 1 else np.mean(dice_scores)
    
    return metrics


def create_segmentation_report(all_metrics: list, 
                              class_labels: np.ndarray,
                              output_dir: Path) -> Dict[str, Any]:
    """
    Creates a summary report and visualizations for segmentation evaluation.
    """
    if not all_metrics:
        return {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = all_metrics[0].keys()
    aggregated = {}
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    # Plot per-class Dice scores
    dice_keys = [k for k in metric_keys if k.startswith('dice_')]
    if dice_keys:
        fig, ax = plt.subplots(figsize=(15, 7))
        dice_means = [aggregated.get(f'{k}_mean', 0) for k in dice_keys]
        dice_stds = [aggregated.get(f'{k}_std', 0) for k in dice_keys]
        labels = [k.replace('dice_class_', 'label_') for k in dice_keys]
        
        ax.bar(range(len(dice_means)), dice_means, yerr=dice_stds, capsize=5)
        ax.set_xlabel('Classes (Label Value)', fontsize=12)
        ax.set_ylabel('Dice Score', fontsize=12)
        ax.set_title('Per-Class Dice Scores', fontsize=14)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dice_scores_per_class.png', dpi=300)
        plt.close()
    
    return aggregated


def main():
    # 1. Configuration and Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "brain_age_pred/configs/multitask/finetune.yaml"
    cfg = Config(cfg_file)
    
    output_dir = Path("output/synthseg_port_evaluation") / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("synthseg-port-eval", log_file=output_dir / "evaluation.log")
    logger.info("=" * 60)
    logger.info("SYNTHSEG PYTORCH PORT EVALUATION")
    logger.info("=" * 60)
    
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")
    
    # 2. Model Setup
    n_classes = len(GENERATION_LABELS)
    logger.info(f"Initializing SynthSeg model with {n_classes} classes.")
    
    # From the prediction script, these are the default parameters for the SynthSeg model
    model_params = {
        'n_levels': 5,
        'n_convs': 2,
        'init_feat': 24,
        'feat_mult': 2,
    }
    model = SynthSeg(n_classes=n_classes, **model_params).to(device)
    
    synthseg_path = cfg.get("synthseg.checkpoint_path")
    if not synthseg_path or not Path(synthseg_path).exists():
        logger.error(f"SynthSeg checkpoint not found at: {synthseg_path}")
        return
        
    logger.info("Loading SynthSeg weights...")
    transfer_summary = load_synthseg_weights(model, synthseg_path, logger)
    stats = transfer_summary.get('transfer_stats', {})
    logger.info(f"Weight transfer complete. Attempted: {stats.get('total_attempted', 'N/A')}, Transferred: {stats.get('successfully_transferred', 'N/A')}")
    
    # 3. Data Setup
    logger.info("Setting up validation data...")
    val_csv = Path(cfg.get("data.val_csv"))
    real_data_dir = Path(cfg.get("data.real_data_dir"))
    
    val_paths, val_ages, val_weights, val_sexes, val_modalities = read_csv(val_csv, real_data_dir)
    logger.info(f"Found {len(val_paths)} validation samples.")
    
    # The ground truth segmentations contain labels from GENERATION_LABELS.
    # We must map these to indices 0, 1, ..., 32 for model training/evaluation.
    output_labels_map = np.arange(len(GENERATION_LABELS))

    validation_generator = ValidationGenerator(
        segmented_data_dir=Path(cfg.get("data.segmented_data_dir", real_data_dir)),
        return_segmentation=True,
        use_intensity_clip_normalize=True,
        generation_labels=GENERATION_LABELS, # The labels present in the GT files
        output_labels=output_labels_map,     # Map GT labels to indices 0-32
    )
    
    val_dataset = BADataset(
        file_paths=val_paths,
        age_labels=val_ages,
        sample_wts=val_weights,
        sexes=val_sexes,
        modalities=val_modalities,
        transform=validation_generator,
        mode="val"
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=cfg.get("data.num_workers", 4),
        pin_memory=cfg.get("data.pin_memory", True)
    )
    
    # 4. Evaluation
    logger.info("Starting evaluation...")
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", leave=True)
        for batch in pbar:
            imgs = batch["image"].to(device)
            seg_gts = batch["seg_gt"].to(device) # Shape: (B, 1, D, H, W), values are 0-32
            
            seg_logits = model(imgs) # Shape: (B, 33, D, H, W)
            
            batch_metrics = compute_segmentation_metrics(
                predictions=seg_logits,
                targets=seg_gts.squeeze(1),
                class_labels=GENERATION_LABELS
            )
            all_metrics.append(batch_metrics)
            pbar.set_postfix({'mean_dice': f'{batch_metrics["mean_dice"]:.3f}'})

    # 5. Results Analysis
    logger.info("Computing final metrics and generating report...")
    summary_stats = create_segmentation_report(all_metrics, GENERATION_LABELS, output_dir)
    
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mean Accuracy: {summary_stats.get('accuracy_mean', 0):.4f}")
    logger.info(f"Mean Dice (excluding background): {summary_stats.get('mean_dice_mean', 0):.4f}")
    
    results = {
        "config": {"config_file": cfg_file, "synthseg_path": str(synthseg_path)},
        "transfer_summary": transfer_summary,
        "summary_statistics": summary_stats,
    }
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x.tolist() if isinstance(x, np.ndarray) else x)
    
    logger.info(f"Detailed results saved to: {results_file}")
    logger.info(f"Visualizations saved in: {output_dir}")
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main() 