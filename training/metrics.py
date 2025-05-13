"""
Metrics for brain age prediction.
"""

import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics for brain age prediction.
    
    Args:
        predictions: Predicted ages
        targets: True ages
        
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are 1D arrays
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    # Calculate brain age delta (BAD)
    brain_age_delta = np.mean(predictions - targets)
    
    # Calculate correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    # Calculate age-specific MAE
    age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
    age_specific_mae = {}
    
    for i in range(len(age_bins) - 1):
        bin_start = age_bins[i]
        bin_end = age_bins[i+1]
        bin_mask = (targets >= bin_start) & (targets < bin_end)
        
        if np.sum(bin_mask) > 0:
            bin_mae = mean_absolute_error(targets[bin_mask], predictions[bin_mask])
            age_specific_mae[f"mae_{bin_start}_{bin_end}"] = bin_mae
    
    # Combine all metrics
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "brain_age_delta": brain_age_delta,
        "correlation": correlation
    }
    
    # Add age-specific MAE
    metrics.update(age_specific_mae)
    
    return metrics
