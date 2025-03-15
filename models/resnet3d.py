"""
3D ResNet for brain age prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from models.base_model import BrainAgeModel


class ResidualBlock3D(nn.Module):
    """3D Residual Block with optional bottleneck architecture."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        bottleneck: bool = False,
        bottleneck_factor: int = 4,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: nn.Module = nn.BatchNorm3d
    ):
        super().__init__()
        
        # Bottleneck architecture
        if bottleneck:
            width = int(out_channels * (base_width / 64.)) * groups
            
            self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, bias=False)
            self.bn1 = norm_layer(width)
            
            self.conv2 = nn.Conv3d(
                width, width, kernel_size=3, stride=stride, padding=dilation,
                groups=groups, bias=False, dilation=dilation
            )
            self.bn2 = norm_layer(width)
            
            self.conv3 = nn.Conv3d(width, out_channels, kernel_size=1, bias=False)
            self.bn3 = norm_layer(out_channels)
            
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
        
        # Basic architecture
        else:
            self.conv1 = nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=stride,
                padding=dilation, groups=groups, bias=False, dilation=dilation
            )
            self.bn1 = norm_layer(out_channels)
            
            self.conv2 = nn.Conv3d(
                out_channels, out_channels, kernel_size=3, padding=dilation,
                groups=groups, bias=False, dilation=dilation
            )
            self.bn2 = norm_layer(out_channels)
            
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
            
        self.bottleneck = bottleneck
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        if self.bottleneck:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            
            out = self.conv3(out)
            out = self.bn3(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(BrainAgeModel):
    """
    3D ResNet for brain age prediction.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
        use_attention: bool = False,
        model_name: str = "resnet3d",
        layers: List[int] = [2, 2, 2, 2],  # ResNet-18
        bottleneck: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None
    ):
        """
        Initialize the ResNet3D model.
        
        Args:
            in_channels: Number of input channels
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanisms
            model_name: Name of the model
            layers: Number of residual blocks in each layer
            bottleneck: Whether to use bottleneck architecture
            groups: Number of groups for group convolution
            width_per_group: Width per group
            replace_stride_with_dilation: Whether to replace stride with dilation
        """
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.layers = layers
        self.bottleneck = bottleneck
        
        if replace_stride_with_dilation is None:
            # Each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.norm_layer = nn.BatchNorm3d
        
        super().__init__(in_channels, dropout_rate, use_attention, model_name)
    
    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        bottleneck: bool = False
    ) -> nn.Sequential:
        """Make a layer with multiple residual blocks."""
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )
        
        layers = []
        # First block with downsample
        layers.append(
            ResidualBlock3D(
                self.inplanes, planes, stride, downsample, bottleneck,
                groups=self.groups, base_width=self.base_width,
                dilation=previous_dilation, norm_layer=norm_layer
            )
        )
        
        self.inplanes = planes
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock3D(
                    self.inplanes, planes, bottleneck=bottleneck,
                    groups=self.groups, base_width=self.base_width,
                    dilation=self.dilation, norm_layer=norm_layer
                )
            )
        
        return nn.Sequential(*layers)
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build the feature extraction part of the model."""
        # Initial layers
        layers = [
            nn.Conv3d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self.norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        ]
        
        # Residual blocks
        layers.append(self._make_layer(64, self.layers[0], bottleneck=self.bottleneck))
        layers.append(self._make_layer(128, self.layers[1], stride=2, 
                                      dilate=self.replace_stride_with_dilation[0],
                                      bottleneck=self.bottleneck))
        layers.append(self._make_layer(256, self.layers[2], stride=2, 
                                      dilate=self.replace_stride_with_dilation[1],
                                      bottleneck=self.bottleneck))
        layers.append(self._make_layer(512, self.layers[3], stride=2, 
                                      dilate=self.replace_stride_with_dilation[2],
                                      bottleneck=self.bottleneck))
        
        # Global pooling
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _build_regression_head(self) -> nn.Module:
        """Build the regression head for age prediction."""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)
        ) 