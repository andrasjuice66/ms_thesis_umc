"""
3D EfficientNet for brain age prediction.
Adapted from the 2D EfficientNet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from models.base_model import BrainAgeModel


class MBConvBlock3D(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block for 3D EfficientNet.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion_factor: int = 6,
        se_ratio: float = 0.25,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the MBConv block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for depthwise convolution
            stride: Stride for depthwise convolution
            expansion_factor: Expansion factor for inverted bottleneck
            se_ratio: Squeeze-and-excitation ratio
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.use_residual = in_channels == out_channels and stride == 1
        self.dropout_rate = dropout_rate
        
        # Expansion phase
        expanded_channels = in_channels * expansion_factor
        self.expand_conv = nn.Sequential(
            nn.Conv3d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(expanded_channels),
            nn.SiLU(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(
                expanded_channels, expanded_channels, kernel_size=kernel_size,
                stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm3d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze and excitation
        self.se = SqueezeExcitation3D(
            expanded_channels, int(in_channels * se_ratio)
        ) if se_ratio > 0 else nn.Identity()
        
        # Projection phase
        self.project_conv = nn.Sequential(
            nn.Conv3d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Save input for residual connection
        residual = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Squeeze and excitation
        x = self.se(x)
        
        # Projection
        x = self.project_conv(x)
        
        # Residual connection
        if self.use_residual:
            if self.training and self.dropout_rate > 0:
                x = F.dropout3d(x, p=self.dropout_rate, training=self.training)
            x = x + residual
        
        return x


class SqueezeExcitation3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D data.
    """
    
    def __init__(self, in_channels: int, reduced_channels: int):
        """
        Initialize the SE block.
        
        Args:
            in_channels: Number of input channels
            reduced_channels: Number of channels after reduction
        """
        super().__init__()
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x * self.se(x)


class EfficientNet3D(BrainAgeModel):
    """
    3D EfficientNet for brain age prediction.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
        use_attention: bool = False,
        model_name: str = "efficientnet3d",
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0
    ):
        """
        Initialize the 3D EfficientNet model.
        
        Args:
            in_channels: Number of input channels
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanisms
            model_name: Name of the model
            width_multiplier: Width multiplier for scaling
            depth_multiplier: Depth multiplier for scaling
        """
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        super().__init__(in_channels, dropout_rate, use_attention, model_name)
    
    def _round_filters(self, filters: int) -> int:
        """Round number of filters based on width multiplier."""
        return int(filters * self.width_multiplier)
    
    def _round_repeats(self, repeats: int) -> int:
        """Round number of repeats based on depth multiplier."""
        return int(repeats * self.depth_multiplier)
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build the feature extraction part of the model."""
        # Initial convolution
        layers = [
            nn.Conv3d(self.in_channels, self._round_filters(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(self._round_filters(32)),
            nn.SiLU(inplace=True)
        ]
        
        # MBConv blocks configuration
        # (expansion_factor, channels, repeats, stride, kernel_size)
        block_configs = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3)
        ]
        
        # Build MBConv blocks
        in_channels = self._round_filters(32)
        for expansion_factor, channels, repeats, stride, kernel_size in block_configs:
            out_channels = self._round_filters(channels)
            repeats = self._round_repeats(repeats)
            
            # First block with stride
            layers.append(
                MBConvBlock3D(
                    in_channels, out_channels, kernel_size, stride,
                    expansion_factor, dropout_rate=self.dropout_rate
                )
            )
            
            # Remaining blocks
            for _ in range(1, repeats):
                layers.append(
                    MBConvBlock3D(
                        out_channels, out_channels, kernel_size, 1,
                        expansion_factor, dropout_rate=self.dropout_rate
                    )
                )
            
            in_channels = out_channels
        
        # Final convolution
        final_channels = self._round_filters(1280)
        layers.extend([
            nn.Conv3d(in_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(final_channels),
            nn.SiLU(inplace=True)
        ])
        
        # Global pooling
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _build_regression_head(self) -> nn.Module:
        """Build the regression head for age prediction."""
        final_channels = self._round_filters(1280)
        
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_channels, 1)
        ) 