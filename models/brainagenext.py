#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Francesco La Rosa
"""
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd
from monai.data import CacheDataset
import numpy as np
import os
import torchio
import torch.nn as nn
import matplotlib.pyplot as plt
from nnunet_mednext import create_mednext_encoder_v1
from models.base_model import BrainAgeModel


class BrainAgeNeXt(nn.Module):
    """
    MedNeXt-based model for brain age prediction.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.0,
        model_id: str = 'B',
        kernel_size: int = 3,
        deep_supervision: bool = True,
        feature_size: int = 512,
        hidden_size: int = 64
    ):
        """
        Initialize the BrainAgeNeXt model.
        
        Args:
            in_channels: Number of input channels
            dropout_rate: Dropout rate
            model_id: MedNeXt model variant (S, B, L, etc.)
            kernel_size: Kernel size for convolutions
            deep_supervision: Whether to use deep supervision
            feature_size: Size of the feature vector from encoder
            hidden_size: Size of the hidden layer in regression head
        """
        super(BrainAgeNeXt, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.model_id = model_id
        self.kernel_size = kernel_size
        self.deep_supervision = deep_supervision
        
        # Build model components
        self.feature_extractor = self._build_feature_extractor()
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regression_head = self._build_regression_head(feature_size, hidden_size)
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build the feature extraction part of the model."""
        return create_mednext_encoder_v1(
            num_input_channels=self.in_channels, 
            num_classes=1, 
            model_id=self.model_id, 
            kernel_size=self.kernel_size, 
            deep_supervision=self.deep_supervision
        )
    
    def _build_regression_head(self, feature_size: int, hidden_size: int) -> nn.Module:
        """Build the regression head for age prediction."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.feature_extractor(x)
        x = self.global_avg_pool(features)
        age_estimate = self.regression_head(x)
        return age_estimate.squeeze()



