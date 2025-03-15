"""
Simple Fully Convolutional Network (SFCN) for brain age prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from models.base_model import BrainAgeModel


class SFCN(BrainAgeModel):
    """
    Simple Fully Convolutional Network for brain age prediction.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
        use_attention: bool = False,
        model_name: str = "sfcn",
        channel_number: List[int] = [32, 64, 128, 256, 256, 64],
        avg_pool_shape: List[int] = [5, 6, 5]
    ):
        """
        Initialize the SFCN model.
        
        Args:
            in_channels: Number of input channels
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanisms
            model_name: Name of the model
            channel_number: List of channel numbers for each layer
            avg_pool_shape: Shape for average pooling
        """
        self.channel_number = channel_number
        self.avg_pool_shape = avg_pool_shape
        super().__init__(in_channels, dropout_rate, use_attention, model_name)
    
    def _conv_layer(
        self, 
        in_channel: int, 
        out_channel: int, 
        maxpool: bool = True, 
        kernel_size: int = 3, 
        padding: int = 0, 
        maxpool_stride: int = 2
    ) -> nn.Sequential:
        """Create a convolutional layer with optional max pooling."""
        if maxpool:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer
    
    def _build_feature_extractor(self) -> nn.Sequential:
        """Build the feature extraction part of the model."""
        n_layer = len(self.channel_number)
        feature_extractor = nn.Sequential()
        
        for i in range(n_layer):
            if i == 0:
                in_channel = self.in_channels
            else:
                in_channel = self.channel_number[i-1]
            
            out_channel = self.channel_number[i]
            
            if i < n_layer-1:
                feature_extractor.add_module(
                    f'conv_{i}',
                    self._conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=True,
                        kernel_size=3,
                        padding=1
                    )
                )
            else:
                feature_extractor.add_module(
                    f'conv_{i}',
                    self._conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=False,
                        kernel_size=1,
                        padding=0
                    )
                )
        
        return feature_extractor
    
    def _build_regression_head(self) -> nn.Sequential:
        """Build the regression head for age prediction."""
        classifier = nn.Sequential()
        
        # Average pooling
        classifier.add_module('average_pool', nn.AvgPool3d(self.avg_pool_shape))
        
        # Dropout
        if self.dropout_rate > 0:
            classifier.add_module('dropout', nn.Dropout(self.dropout_rate))
        
        # Final convolution to get age prediction
        classifier.add_module(
            f'conv_{len(self.channel_number)}',
            nn.Conv3d(self.channel_number[-1], 1, padding=0, kernel_size=1)
        )
        
        # Flatten to get final output
        classifier.add_module('flatten', nn.Flatten())
        
        return classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.feature_extractor(x)
        age = self.regression_head(features)
        return age.squeeze()


