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
    MSE loss with sample-specific weights.
    Can use either auto-generated weights based on age extremity or external weights.
    """
    
    def __init__(self, min_age: float = 0.0, max_age: float = 100.0, alpha: float = 1.0):
        """
        Initialize the loss function.
        
        Args:
            min_age: Minimum expected age
            max_age: Maximum expected age
            alpha: Weight factor for auto-generated weights
        """
        super().__init__()
        self.min_age = min_age
        self.max_age = max_age
        self.alpha = alpha
        self.range = max_age - min_age
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the loss.
        
        Args:
            pred: Predicted values
            target: Target values
            weights: Optional sample-specific weights
            
        Returns:
            Loss value
        """
        # Calculate MSE without reduction
        squared_error = (pred - target) ** 2
        
        if weights is None:
            # Use auto-generated weights based on age extremity
            normalized_target = (target - self.min_age) / self.range
            weights = 1.0 + self.alpha * (2.0 * torch.abs(normalized_target - 0.5))
        
        # Apply weights to squared errors
        weighted_squared_error = weights * squared_error
        
        return torch.mean(weighted_squared_error)


class WeightedMAELoss(nn.Module):
    """
    MAE loss with sample-specific weights.
    Can use either auto-generated weights based on age extremity or external weights.
    """
    
    def __init__(self, min_age: float = 0.0, max_age: float = 100.0, alpha: float = 1.0):
        """
        Initialize the loss function.
        
        Args:
            min_age: Minimum expected age
            max_age: Maximum expected age
            alpha: Weight factor for auto-generated weights
        """
        super().__init__()
        self.min_age = min_age
        self.max_age = max_age
        self.alpha = alpha
        self.range = max_age - min_age
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the loss.
        
        Args:
            pred: Predicted values
            target: Target values
            weights: Optional sample-specific weights
            
        Returns:
            Loss value
        """
        # Calculate MAE without reduction
        absolute_error = torch.abs(pred - target)
        
        if weights is None:
            # Use auto-generated weights based on age extremity
            normalized_target = (target - self.min_age) / self.range
            weights = 1.0 + self.alpha * (2.0 * torch.abs(normalized_target - 0.5))
        
        # Apply weights to absolute errors
        weighted_absolute_error = weights * absolute_error
        
        return torch.mean(weighted_absolute_error)


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss for comparing probability distributions.
    Assumes input is log-probabilities and target is probabilities.
    """
    def __init__(self, reduction: str = 'sum'):
        """
        Initialize the loss function.

        Args:
            reduction: Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'
        """
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the KL divergence loss.

        Args:
            input: Log-probabilities (output of log_softmax)
            target: Probabilities (output of softmax or one-hot)

        Returns:
            Loss value
        """
        return self.kl_div(input, target)


class CustomKLDivergenceLoss(nn.Module):
    """
    Custom KL Divergence loss that:
    1. Averages by batch size (0th dimension)
    2. Adds small epsilon to target distribution to prevent log(0)
    """
    def __init__(self, epsilon: float = 1e-16):
        """
        Initialize the loss function.

        Args:
            epsilon: Small value to add to target distribution to prevent log(0)
        """
        super().__init__()
        self.epsilon = epsilon
        self.kl_div = nn.KLDivLoss(reduction='sum')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the custom KL divergence loss.

        Args:
            input: Log-probabilities (output of log_softmax)
            target: Probabilities (output of softmax or one-hot)

        Returns:
            Loss value averaged by batch size
        """
        # Add small epsilon to prevent log(0)
        target = target + self.epsilon
        # Calculate loss and average by batch size
        n = target.shape[0]
        loss = self.kl_div(input, target) / n
        return loss


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
        ),
        "weighted_mae": WeightedMAELoss(
            min_age=kwargs.get("min_age", 0.0),
            max_age=kwargs.get("max_age", 100.0),
            alpha=kwargs.get("alpha", 1.0)
        ),
        # "kl_div": KLDivergenceLoss(
        #     reduction=kwargs.get("reduction", "batchmean")
        # ),
        "kl_div": CustomKLDivergenceLoss(
            epsilon=kwargs.get("epsilon", 1e-16)
        ),
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_type}")
    
    return loss_functions[loss_type]
