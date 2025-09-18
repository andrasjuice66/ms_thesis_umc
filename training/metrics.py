"""
Metrics for brain age prediction.
"""

import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray,
    modalities: Optional[List[str]] = None,
    sexes: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate regression metrics for brain age prediction.
    
    Args:
        predictions: Predicted ages
        targets: True ages
        modalities: List of modalities for each sample (optional)
        sexes: List of sexes for each sample (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are 1D arrays
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    
    # DEBUG: Check for NaN values with detailed info
    if np.isnan(predictions).any():
        nan_count = np.isnan(predictions).sum()
        total_count = len(predictions)
        nan_indices = np.where(np.isnan(predictions))[0]
        
        print(f"WARNING: Found {nan_count}/{total_count} NaN values in predictions!")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions min: {np.nanmin(predictions)}, max: {np.nanmax(predictions)}")
        print(f"Non-NaN predictions range: {predictions[~np.isnan(predictions)][:10] if nan_count < total_count else 'All NaN!'}")
        
        # Show which specific indices/samples have NaN values
        print(f"NaN prediction indices: {nan_indices}")
        print(f"First 10 NaN indices: {nan_indices[:10]}")
        
        # Show corresponding target values for NaN predictions
        print(f"Target values for NaN predictions: {targets[nan_indices]}")
        
        # If modalities are provided, show which modalities have NaN predictions
        if modalities is not None:
            nan_modalities = [modalities[i] for i in nan_indices]
            print(f"Modalities with NaN predictions: {nan_modalities}")
            from collections import Counter
            modality_counts = Counter(nan_modalities)
            print(f"NaN count by modality: {dict(modality_counts)}")
        
        # If sexes are provided, show sex distribution of NaN predictions
        if sexes is not None:
            nan_sexes = [sexes[i] for i in nan_indices]
            print(f"Sexes with NaN predictions: {nan_sexes}")
            from collections import Counter
            sex_counts = Counter(nan_sexes)
            print(f"NaN count by sex: {dict(sex_counts)}")
            
        # FIX: Instead of raising an error, replace NaNs with mean prediction
        # to allow training to continue
        mean_pred = np.nanmean(predictions)
        if np.isnan(mean_pred):  # If all predictions are NaN
            mean_pred = np.mean(targets)  # Use target mean as fallback
        
        print(f"Replacing {nan_count} NaN predictions with mean value: {mean_pred:.4f}")
        predictions[nan_indices] = mean_pred
        
        # Return warning instead of error
        print(f"WARNING: {nan_count} NaN predictions were replaced with mean value {mean_pred:.4f}")
    
    if np.isnan(targets).any():
        nan_count = np.isnan(targets).sum()
        print(f"ERROR: Found {nan_count} NaN values in targets!")
        raise ValueError(f"Targets contain {nan_count} NaN values")
    
    # Calculate overall metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    brain_age_delta = np.mean(predictions - targets)
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
    
    # Calculate modality-specific metrics if modalities are provided
    if modalities is not None:
        unique_modalities = np.unique(modalities)
        for modality in unique_modalities:
            mask = np.array(modalities) == modality
            if np.sum(mask) > 0:
                mod_mae = mean_absolute_error(targets[mask], predictions[mask])
                mod_mse = mean_squared_error(targets[mask], predictions[mask])
                mod_rmse = np.sqrt(mod_mse)
                mod_r2 = r2_score(targets[mask], predictions[mask])
                mod_delta = np.mean(predictions[mask] - targets[mask])
                mod_corr = np.corrcoef(predictions[mask], targets[mask])[0, 1]
                
                metrics.update({
                    f"{modality}_mae": mod_mae,
                    f"{modality}_mse": mod_mse,
                    f"{modality}_rmse": mod_rmse,
                    f"{modality}_r2": mod_r2,
                    f"{modality}_brain_age_delta": mod_delta,
                    f"{modality}_correlation": mod_corr
                })
    
    # Calculate sex-specific metrics if sexes are provided
    if sexes is not None:
        unique_sexes = np.unique(sexes)
        for sex in unique_sexes:
            mask = np.array(sexes) == sex
            if np.sum(mask) > 0:
                sex_mae = mean_absolute_error(targets[mask], predictions[mask])
                sex_mse = mean_squared_error(targets[mask], predictions[mask])
                sex_rmse = np.sqrt(sex_mse)
                sex_r2 = r2_score(targets[mask], predictions[mask])
                sex_delta = np.mean(predictions[mask] - targets[mask])
                sex_corr = np.corrcoef(predictions[mask], targets[mask])[0, 1]
                
                metrics.update({
                    f"{sex}_mae": sex_mae,
                    f"{sex}_mse": sex_mse,
                    f"{sex}_rmse": sex_rmse,
                    f"{sex}_r2": sex_r2,
                    f"{sex}_brain_age_delta": sex_delta,
                    f"{sex}_correlation": sex_corr
                })
    
    return metrics
