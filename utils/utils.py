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

def load_checkpoint(model, checkpoint_path, device, logger):
    """
    Load model checkpoint with proper error handling and logging.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        logger: Logger instance for logging messages
    
    Returns:
        dict: Additional checkpoint information (epoch, optimizer state, etc.) if available
    """
    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Full checkpoint with state dict and other info
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
                
            # Remove 'module.' prefix if it exists (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Filter out incompatible InstanceNorm3d running stats buffers
            # These were created in older PyTorch versions where track_running_stats=True by default
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            incompatible_keys = []
            
            for k, v in new_state_dict.items():
                # Check if this key exists in the current model
                if k in model_state_dict:
                    # Check if shapes match
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        incompatible_keys.append(f"{k}: shape mismatch ({v.shape} vs {model_state_dict[k].shape})")
                else:
                    # Key doesn't exist in current model - could be running_mean/running_var from InstanceNorm3d
                    if 'running_mean' in k or 'running_var' in k:
                        incompatible_keys.append(f"{k}: InstanceNorm3d running stats (track_running_stats=False in current model)")
                    else:
                        incompatible_keys.append(f"{k}: missing in current model")
            
            if incompatible_keys:
                logger.warning(f"Skipping {len(incompatible_keys)} incompatible keys:")
                for key in incompatible_keys:
                    logger.warning(f"  - {key}")
                
            # Load the filtered state dict
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"Successfully loaded model weights ({len(filtered_state_dict)}/{len(new_state_dict)} keys)")
            
            # Return additional checkpoint info if available
            return {
                'epoch': checkpoint.get('epoch'),
                'optimizer_state': checkpoint.get('optimizer_state_dict'),
                'scheduler_state': checkpoint.get('scheduler_state_dict'),
                'best_metric': checkpoint.get('best_metric'),
                'history': checkpoint.get('history')
            }
        else:
            # Assume the checkpoint is just the state dict
            # Apply same filtering for this case
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            incompatible_keys = []
            
            for k, v in checkpoint.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        incompatible_keys.append(f"{k}: shape mismatch ({v.shape} vs {model_state_dict[k].shape})")
                else:
                    if 'running_mean' in k or 'running_var' in k:
                        incompatible_keys.append(f"{k}: InstanceNorm3d running stats (track_running_stats=False in current model)")
                    else:
                        incompatible_keys.append(f"{k}: missing in current model")
            
            if incompatible_keys:
                logger.warning(f"Skipping {len(incompatible_keys)} incompatible keys:")
                for key in incompatible_keys:
                    logger.warning(f"  - {key}")
            
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"Successfully loaded model weights ({len(filtered_state_dict)}/{len(checkpoint)} keys)")
            return {}
            
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise



# Add this function before the main() function in train_seg_map.py
def load_checkpoint_with_different_channels(model, checkpoint_path, device, logger, 
                                           original_in_channels=1, new_in_channels=15):
    """
    Load checkpoint for a model with different input channels.
    
    This function specifically handles the case where we're loading weights from a model
    trained on different number of input channels. It will:
    1. Load the checkpoint
    2. Skip the first layer weights that would have shape mismatch
    3. Apply all other weights that match
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        logger: Logger instance
        original_in_channels: Number of input channels in the checkpoint
        new_in_channels: Number of input channels in the current model
    """
    try:
        logger.info(f"Loading checkpoint with channel adaptation from {checkpoint_path}")
        logger.info(f"Adapting weights from {original_in_channels} to {new_in_channels} channels")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Get the state dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if it exists (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # Get the current model state dict
        model_state_dict = model.state_dict()
        
        # These are the keys for the first convolution layer that need special handling
        stem_keys = [k for k in new_state_dict.keys() if 'stem.weight' in k]
        
        # Initialize statistics
        total_layers = len(new_state_dict)
        layers_loaded = 0
        layers_skipped = 0
        
        # Process the state dict
        filtered_state_dict = {}
        
        for k, v in new_state_dict.items():
            # Skip the stem layer with shape mismatch
            if k in stem_keys:
                logger.warning(f"Skipping first layer weights due to channel mismatch: {k} {v.shape} vs {model_state_dict[k].shape}")
                layers_skipped += 1
                continue
                
            # Check if the key exists and shapes match
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                    layers_loaded += 1
                else:
                    logger.warning(f"Shape mismatch for {k}: {v.shape} vs {model_state_dict[k].shape}")
                    layers_skipped += 1
            else:
                layers_skipped += 1
                
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        logger.info(f"Successfully loaded model weights with channel adaptation")
        logger.info(f"Loaded {layers_loaded}/{total_layers} layers, skipped {layers_skipped} layers")
        logger.info(f"The input stem layer was initialized randomly")
        
        # Return additional checkpoint info
        if isinstance(checkpoint, dict):
            return {
                'epoch': checkpoint.get('epoch'),
                'optimizer_state': checkpoint.get('optimizer_state_dict'),
                'scheduler_state': checkpoint.get('scheduler_state_dict'),
                'best_metric': checkpoint.get('best_metric'),
                'history': checkpoint.get('history')
            }
        return {}
        
    except Exception as e:
        logger.error(f"Error loading checkpoint with channel adaptation: {str(e)}")
        raise
