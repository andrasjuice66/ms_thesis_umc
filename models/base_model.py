"""
Base model class for brain age prediction.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json


class BrainAgeModel(nn.Module, ABC):
    """
    Base class for brain age prediction models.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
        use_attention: bool = False,
        model_name: str = "base_model"
    ):
        """
        Initialize the brain age model.
        
        Args:
            in_channels: Number of input channels
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanisms
            model_name: Name of the model
        """
        super().__init__()
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.model_name = model_name
        
        # Build the model
        self.feature_extractor = self._build_feature_extractor()
        self.regression_head = self._build_regression_head()
        
        # Initialize metrics tracking
        self.best_metric = float('inf')
        self.current_epoch = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": []
        }
    
    @abstractmethod
    def _build_feature_extractor(self) -> nn.Module:
        """Build the feature extraction part of the model."""
        pass
    
    @abstractmethod
    def _build_regression_head(self) -> nn.Module:
        """Build the regression head for age prediction."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.feature_extractor(x)
        age = self.regression_head(features)
        return age
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature maps for visualization."""
        return self.feature_extractor(x)
    
    def save_checkpoint(self, checkpoint_dir: str, metrics: Dict[str, float], extra_info: Optional[Dict[str, Any]] = None):
        """
        Save model checkpoint with metrics and additional information.
        
        Args:
            checkpoint_dir: Directory to save the checkpoint
            metrics: Dictionary of metrics to save
            extra_info: Additional information to save
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Update training history
        for key, value in metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
        
        # Check if this is the best model
        current_metric = metrics.get("val_mae", float('inf'))
        is_best = current_metric < self.best_metric
        
        if is_best:
            self.best_metric = current_metric
        
        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "epoch": self.current_epoch,
            "metrics": metrics,
            "training_history": self.training_history,
            "best_metric": self.best_metric,
            "model_config": {
                "in_channels": self.in_channels,
                "dropout_rate": self.dropout_rate,
                "use_attention": self.use_attention
            }
        }
        
        # Add extra info if provided
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.model_name}_epoch_{self.current_epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(checkpoint_dir, f"{self.model_name}_best.pth")
            torch.save(checkpoint, best_path)
        
        # Save training history as JSON
        history_path = os.path.join(checkpoint_dir, f"{self.model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        
        self.current_epoch += 1
        
        return checkpoint_path, is_best
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, map_location: Optional[torch.device] = None):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            map_location: Device to map the model to
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Get model configuration
        model_config = checkpoint.get("model_config", {})
        
        # Create model instance
        model = cls(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load training history and metrics
        model.training_history = checkpoint.get("training_history", model.training_history)
        model.best_metric = checkpoint.get("best_metric", model.best_metric)
        model.current_epoch = checkpoint.get("epoch", 0) + 1  # Increment for next epoch
        
        return model
