#!/usr/bin/env python
"""
Test script to load SynthSeg weights and evaluate segmentation performance on validation data.
"""
import os, sys, time, json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import multiprocessing as mp

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.metrics import DiceMetric
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.dataset.dataset import BADataset
from brain_age_pred.models.multi_head import MultiTaskBrainAge
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv
from brain_age_pred.utils.weight_transfer import transfer_synthseg_weights
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


def compute_segmentation_metrics(predictions: torch.Tensor, 
                                targets: torch.Tensor, 
                                num_classes: int,
                                class_names: Optional[list] = None) -> Dict[str, float]:
    """
    Compute comprehensive segmentation metrics.
    
    Args:
        predictions: Model predictions (B, C, D, H, W) or (B, D, H, W)
        targets: Ground truth labels (B, D, H, W)
        num_classes: Number of classes
        class_names: Optional list of class names
        
    Returns:
        Dictionary containing various segmentation metrics
    """
    # Convert to numpy and ensure correct shapes
    if predictions.dim() == 5:  # (B, C, D, H, W) - logits
        preds = torch.argmax(predictions, dim=1)  # (B, D, H, W)
    else:  # Already argmax'd
        preds = predictions
    
    preds = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    # Remove any remaining out-of-bounds values
    valid_mask = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = np.mean(preds == targets)
    
    # Per-class Dice coefficients
    dice_scores = []
    iou_scores = []
    
    for class_id in range(num_classes):
        pred_mask = (preds == class_id)
        target_mask = (targets == class_id)
        
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)
        pred_sum = np.sum(pred_mask)
        target_sum = np.sum(target_mask)
        
        # Dice coefficient
        if pred_sum + target_sum > 0:
            dice = 2.0 * intersection / (pred_sum + target_sum)
        else:
            dice = 1.0  # Perfect score when both are empty
        
        # IoU (Jaccard index)
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0  # Perfect score when both are empty
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        
        # Store per-class metrics
        class_name = class_names[class_id] if class_names else f"class_{class_id}"
        metrics[f'dice_{class_name}'] = dice
        metrics[f'iou_{class_name}'] = iou
        metrics[f'pixels_{class_name}'] = target_sum
    
    # Mean metrics (excluding background if class 0)
    metrics['mean_dice'] = np.mean(dice_scores[1:] if num_classes > 1 else dice_scores)
    metrics['mean_iou'] = np.mean(iou_scores[1:] if num_classes > 1 else iou_scores)
    metrics['mean_dice_all'] = np.mean(dice_scores)
    metrics['mean_iou_all'] = np.mean(iou_scores)
    
    return metrics


def create_segmentation_report(all_metrics: list, 
                              class_names: Optional[list] = None,
                              output_dir: Path = None) -> Dict[str, Any]:
    """
    Create a comprehensive segmentation evaluation report.
    
    Args:
        all_metrics: List of metric dictionaries from each batch
        class_names: Optional list of class names
        output_dir: Directory to save visualizations
        
    Returns:
        Summary statistics dictionary
    """
    if not all_metrics:
        return {}
    
    # Aggregate metrics across all batches
    metric_keys = all_metrics[0].keys()
    aggregated = {}
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
    
    # Create visualizations if output directory is provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot per-class Dice scores
        dice_keys = [k for k in metric_keys if k.startswith('dice_') and not k.endswith('_all')]
        if dice_keys:
            fig, ax = plt.subplots(figsize=(12, 6))
            dice_means = [aggregated[f'{k}_mean'] for k in dice_keys]
            dice_stds = [aggregated[f'{k}_std'] for k in dice_keys]
            class_labels = [k.replace('dice_', '') for k in dice_keys]
            
            bars = ax.bar(range(len(dice_means)), dice_means, yerr=dice_stds, capsize=5)
            ax.set_xlabel('Classes')
            ax.set_ylabel('Dice Score')
            ax.set_title('Per-Class Dice Scores')
            ax.set_xticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, dice_means, dice_stds)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'dice_scores_per_class.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot IoU scores
        iou_keys = [k for k in metric_keys if k.startswith('iou_') and not k.endswith('_all')]
        if iou_keys:
            fig, ax = plt.subplots(figsize=(12, 6))
            iou_means = [aggregated[f'{k}_mean'] for k in iou_keys]
            iou_stds = [aggregated[f'{k}_std'] for k in iou_keys]
            class_labels = [k.replace('iou_', '') for k in iou_keys]
            
            bars = ax.bar(range(len(iou_means)), iou_means, yerr=iou_stds, capsize=5, color='orange', alpha=0.7)
            ax.set_xlabel('Classes')
            ax.set_ylabel('IoU Score')
            ax.set_title('Per-Class IoU Scores')
            ax.set_xticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, iou_means, iou_stds)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'iou_scores_per_class.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    return aggregated


def main():
    # 1. Configuration & Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/multitask/finetune.yaml"
    cfg = Config(cfg_file)
    
    # Output directory
    output_dir = Path("output/synthseg_evaluation") / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = setup_logger("synthseg-eval", log_file=output_dir / "evaluation.log")
    
    logger.info("=" * 60)
    logger.info("SYNTHSEG SEGMENTATION EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Config file: {cfg_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set seed for reproducibility
    set_seed(cfg.get("seed", 42))
    
    # Device
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")
    
    # 2. Model Setup
    n_classes = 33  # SynthSeg classes (0-32)
    logger.info(f"Initializing MultiTaskBrainAge model with {n_classes} classes")
    model = MultiTaskBrainAge(n_classes=n_classes).to(device)
    
    # Load SynthSeg weights
    synthseg_cfg = cfg.get("synthseg", {})
    synthseg_path = synthseg_cfg.get("checkpoint_path")
    
    if not synthseg_path or not Path(synthseg_path).exists():
        logger.error(f"SynthSeg checkpoint not found: {synthseg_path}")
        logger.error("Please specify a valid 'synthseg.checkpoint_path' in your config file")
        return
    
    logger.info("Loading SynthSeg weights...")
    try:
        transfer_summary = load_synthseg_weights(
            model=model,
            synthseg_path=synthseg_path,
            freeze_encoder=synthseg_cfg.get("freeze_encoder", False),
            freeze_decoder=synthseg_cfg.get("freeze_decoder", False),
            logger=logger
        )
        
        # Log transfer statistics
        stats = transfer_summary['transfer_stats']
        logger.info(f"Transfer complete: {stats['successfully_transferred']}/{stats['total_attempted']} layers")
        logger.info(f"Transfer rate: {stats['successfully_transferred']/stats['total_attempted']*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Failed to load SynthSeg weights: {e}")
        return
    
    # 3. Data Setup
    logger.info("Setting up validation data...")
    
    # Read validation CSV
    val_csv = Path(cfg.get("data.val_csv"))
    real_data_dir = Path(cfg.get("data.real_data_dir"))
    
    if not val_csv.exists():
        logger.error(f"Validation CSV not found: {val_csv}")
        return
    
    logger.info(f"Reading validation CSV from {val_csv}")
    val_paths, val_ages, val_weights, val_sexes, val_modalities = read_csv(val_csv, real_data_dir)
    logger.info(f"Found {len(val_paths)} validation samples")
    
    # Create validation generator
    validation_generator = ValidationGenerator(
        segmented_data_dir=Path(cfg.get("data.segmented_data_dir", real_data_dir)),
        return_segmentation=True,
        use_intensity_clip_normalize=True,
        generation_labels=GENERATION_LABELS,     # Input labels (up to 60)
        output_labels=GENERATION_CLASSES,    # Output labels (0-32)
    )
    
    # Create dataset and dataloader
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
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("data.num_workers", 4),
        pin_memory=cfg.get("data.pin_memory", True)
    )
    
    # 4. Evaluation
    logger.info("Starting segmentation evaluation...")
    model.eval()
    
    all_metrics = []
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating segmentation", leave=True)
        for batch_idx, batch in enumerate(pbar):
            imgs = batch["image"].to(device)
            seg_gts = batch["seg_gt"].to(device)
            
            # Forward pass
            seg_logits, age_preds = model(imgs)
            
            # Compute segmentation predictions
            seg_preds = torch.argmax(seg_logits, dim=1, keepdim=True)
            
            # One-hot encode the ground truth (already mapped to 0-32 by data loader)
            try:
                seg_gts_one_hot = F.one_hot(
                    seg_gts.squeeze(1).long(), 
                    num_classes=n_classes
                ).permute(0, 4, 1, 2, 3)
                
                # Update MONAI Dice metric
                dice_metric(y_pred=seg_preds, y=seg_gts_one_hot)
                
            except Exception as e:
                logger.warning(f"Batch {batch_idx}: MONAI Dice metric failed: {e}")
                # Log label range for debugging
                logger.warning(f"Seg GT range for failed batch: min={seg_gts.min().item()}, max={seg_gts.max().item()}")
                continue
            
            # Compute detailed metrics for this batch
            try:
                batch_metrics = compute_segmentation_metrics(
                    predictions=seg_logits,
                    targets=seg_gts.squeeze(1),
                    num_classes=n_classes,
                    class_names=[f"class_{i}" for i in range(n_classes)]
                )
                
                all_metrics.append(batch_metrics)
                
            except Exception as e:
                logger.warning(f"Batch {batch_idx}: Metrics computation failed: {e}")
                continue
            
            total_samples += imgs.shape[0]
            
            # Update progress bar
            try:
                current_dice = dice_metric.aggregate().item()
                pbar.set_postfix({
                    'samples': total_samples,
                    'dice': f'{current_dice:.3f}',
                    'acc': f'{batch_metrics.get("accuracy", 0):.3f}'
                })
            except:
                pbar.set_postfix({'samples': total_samples})
    
    # 5. Results Analysis
    logger.info("Computing final metrics...")
    
    # Get final MONAI Dice score
    try:
        final_dice = dice_metric.aggregate().item()
        logger.info(f"Final Dice Score (MONAI): {final_dice:.4f}")
    except:
        final_dice = float('nan')
        logger.warning("Could not compute final MONAI Dice score")
    
    # Create comprehensive report
    summary_stats = create_segmentation_report(
        all_metrics=all_metrics,
        class_names=[f"class_{i}" for i in range(n_classes)],
        output_dir=output_dir
    )
    
    # Log key metrics
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples evaluated: {total_samples}")
    logger.info(f"Successful batches: {len(all_metrics)}")
    logger.info(f"Overall Dice Score: {final_dice:.4f}")
    logger.info(f"Mean Accuracy: {summary_stats.get('accuracy_mean', 0):.4f} ± {summary_stats.get('accuracy_std', 0):.4f}")
    logger.info(f"Mean Dice (excl. background): {summary_stats.get('mean_dice_mean', 0):.4f} ± {summary_stats.get('mean_dice_std', 0):.4f}")
    logger.info(f"Mean IoU (excl. background): {summary_stats.get('mean_iou_mean', 0):.4f} ± {summary_stats.get('mean_iou_std', 0):.4f}")
    
    # Save detailed results
    results = {
        "evaluation_config": {
            "config_file": cfg_file,
            "synthseg_path": str(synthseg_path),
            "n_classes": n_classes,
            "total_samples": total_samples,
            "successful_batches": len(all_metrics),
            "timestamp": timestamp
        },
        "transfer_summary": transfer_summary,
        "final_dice_score": final_dice,
        "summary_statistics": summary_stats,
        "per_batch_metrics": all_metrics
    }
    
    # Save to JSON
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info(f"Detailed results saved to: {results_file}")
    logger.info(f"Visualizations saved to: {output_dir}")
    
    # Print top performing classes
    dice_keys = [k for k in summary_stats.keys() if k.startswith('dice_class_') and k.endswith('_mean')]
    if dice_keys:
        logger.info("\nTop 5 performing classes (by Dice score):")
        class_performances = [(k, summary_stats[k]) for k in dice_keys]
        class_performances.sort(key=lambda x: x[1], reverse=True)
        
        for i, (class_key, dice_score) in enumerate(class_performances[:5]):
            class_name = class_key.replace('dice_', '').replace('_mean', '')
            logger.info(f"  {i+1}. {class_name}: {dice_score:.4f}")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main() 