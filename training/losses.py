"""
Loss functions for brain age prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable


class HuberMAELoss(nn.Module):
    """
    Combination of Huber loss and MAE loss for robust regression.
    """
    
    def __init__(self, delta: float = 1.0, mae_weight: float = 0.5):
        """
        Initialize the loss function.
        
        Args:
            delta: Threshold for Huber loss
            mae_weight: Weight for MAE loss component
        """
        super().__init__()
        self.delta = delta
        self.mae_weight = mae_weight
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
        self.mae = nn.L1Loss(reduction='mean')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Loss value
        """
        huber_loss = self.huber(pred, target)
        mae_loss = self.mae(pred, target)
        
        return (1 - self.mae_weight) * huber_loss + self.mae_weight * mae_loss


class WeightedMSELoss(nn.Module):
    """
    MSE loss with age-dependent weighting.
    Gives higher weight to samples with extreme ages (young or old).
    """
    
    def __init__(self, min_age: float = 0.0, max_age: float = 100.0, alpha: float = 1.0):
        """
        Initialize the loss function.
        
        Args:
            min_age: Minimum expected age
            max_age: Maximum expected age
            alpha: Weight factor
        """
        super().__init__()
        self.min_age = min_age
        self.max_age = max_age
        self.alpha = alpha
        self.range = max_age - min_age
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Loss value
        """
        # Normalize ages to [0, 1]
        normalized_target = (target - self.min_age) / self.range
        
        # Calculate weights: higher for extreme ages
        weights = 1.0 + self.alpha * (2.0 * torch.abs(normalized_target - 0.5))
        
        # Calculate MSE with weights
        squared_error = (pred - target) ** 2
        weighted_squared_error = weights * squared_error
        
        return torch.mean(weighted_squared_error)


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Get the specified loss function.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function
    """
    loss_functions = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.HuberLoss(delta=kwargs.get("delta", 1.0)),
        "huber_mae": HuberMAELoss(
            delta=kwargs.get("delta", 1.0),
            mae_weight=kwargs.get("mae_weight", 0.5)
        ),
        "weighted_mse": WeightedMSELoss(
            min_age=kwargs.get("min_age", 0.0),
            max_age=kwargs.get("max_age", 100.0),
            alpha=kwargs.get("alpha", 1.0)
        )
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_type}")
    
    return loss_functions[loss_type]
