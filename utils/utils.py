"""
Utility functions for brain age prediction.
"""

import random
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
import pandas as pd

def set_seed(seed):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ───────────────────── helpers ────────────────────── #
def read_csv(
    csv_path: str,
    data_root: str,
    image_key: str = "image_path",
    age_key: str = "age",
    weight_key: str = "sample_weight",
    sex_key: str = "sex",
    modalities_key: str = "modality",
) -> Tuple[List[str], List[float], List[float], List[str], List[str]]:
    df = pd.read_csv(csv_path)
    paths, ages, weights, sexes, modalities = [], [], [], [], []
    data_root = Path(data_root)  # Essure data_root is a Path object
    for _, row in df.iterrows():
        rel_path = row[image_key]
        fpath = data_root / rel_path
        #print(f"Checking: {fpath}")
        if fpath.exists():
            paths.append(str(fpath))
            ages.append(float(row[age_key]))
            weights.append(float(row.get(weight_key, 1.0)))
            sexes.append(str(row.get(sex_key, 'N/A')))
            modalities.append(str(row.get(modalities_key, 'N/A')))
    return paths, ages, weights, sexes, modalities

def load_checkpoint_with_different_channels(
    model,
    checkpoint_path,
    device,
    logger,
    original_in_channels=1,
    new_in_channels=15,
    adapt_first_conv=True,
):
    """
    Load a checkpoint into a model when input channels differ.
    - Adapts the first Conv3d weight (repeat/average) instead of skipping it.
    - Loads all other matching weights as usual.
    """
    try:
        logger.info(f"Loading checkpoint with channel adaptation from {checkpoint_path}")
        logger.info(f"Adapting weights from {original_in_channels} to {new_in_channels} channels")

        # Safe torch.load with optional weights_only (newer PyTorch)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        clean_sd = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            clean_sd[name] = v

        model_sd = model.state_dict()
        adapted_sd = dict(model_sd)  # start from model's current params

        total_layers = len(clean_sd)
        layers_loaded = 0
        layers_skipped = 0

        # Helper to detect and adapt first conv with input-channel mismatch
        def maybe_adapt_first_conv(k, w_pre, w_new):
            # Conv3d: [C_out, C_in, kD, kH, kW]
            if w_pre.ndim == 5 and w_new.ndim == 5:
                c_out_pre, c_in_pre, kd, kh, kw = w_pre.shape
                c_out_new, c_in_new, kd2, kh2, kw2 = w_new.shape
                if (c_out_pre == c_out_new) and (kd == kd2) and (kh == kh2) and (kw == kw2) and (c_in_pre != c_in_new):
                    # Strategy: average across old input channels then repeat to new_in_channels
                    # - If c_in_pre == 1, this is equivalent to repeat
                    w_mean = w_pre.mean(dim=1, keepdim=True)  # [C_out, 1, kD, kH, kW]
                    w_rep = w_mean.repeat(1, c_in_new, 1, 1, 1)
                    return w_rep
            return None

        # First pass: copy everything that matches; adapt first conv if requested
        for k, v in clean_sd.items():
            if k not in model_sd:
                layers_skipped += 1
                continue

            tgt = model_sd[k]
            if v.shape == tgt.shape:
                adapted_sd[k] = v
                layers_loaded += 1
                continue

            if adapt_first_conv and v.ndim == 5 and tgt.ndim == 5:
                w_rep = maybe_adapt_first_conv(k, v, tgt)
                if w_rep is not None:
                    adapted_sd[k] = w_rep
                    layers_loaded += 1
                    logger.info(f"Adapted first conv weights for {k}: {v.shape} -> {tgt.shape}")
                    continue

            logger.warning(f"Shape mismatch for {k}: {v.shape} vs {tgt.shape} (skipping)")
            layers_skipped += 1

        missing, unexpected = model.load_state_dict(adapted_sd, strict=False)
        logger.info("Successfully loaded model weights with channel adaptation")
        logger.info(f"Loaded {layers_loaded}/{total_layers} tensors, skipped {layers_skipped}")
        if missing:
            logger.info(f"Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            logger.info(f"Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

        # Return ancillary training info if present
        if isinstance(checkpoint, dict):
            return {
                'epoch': checkpoint.get('epoch'),
                'optimizer_state': checkpoint.get('optimizer_state_dict'),
                'scheduler_state': checkpoint.get('scheduler_state_dict'),
                'best_metric': checkpoint.get('best_metric'),
                'history': checkpoint.get('history'),
            }
        return {}

    except Exception as e:
        logger.error(f"Error loading checkpoint with channel adaptation: {str(e)}")
        raise