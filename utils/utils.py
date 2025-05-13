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
) -> Tuple[List[str], List[float], List[float]]:
    df = pd.read_csv(csv_path)
    paths, ages, weights = [], [], []
    data_root = Path(data_root)  # Ensure data_root is a Path object
    for _, row in df.iterrows():
        rel_path = row[image_key]
        fpath = data_root / rel_path
        #print(f"Checking: {fpath}")
        if fpath.exists():
            paths.append(str(fpath))
            ages.append(float(row[age_key]))
            weights.append(float(row[weight_key]))
    return paths, ages, weights

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
                
            # Load the state dict
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("Successfully loaded model weights")
            
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
            model.load_state_dict(checkpoint, strict=False)
            logger.info("Successfully loaded model weights (state dict only)")
            return {}
            
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise

