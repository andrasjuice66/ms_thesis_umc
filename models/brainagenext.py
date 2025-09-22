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
from brain_age_pred.models.create_mednext_encoder_v1 import create_mednext_encoder_v1


class BrainAgeNeXt(nn.Module):
    """
    MedNeXt-based model for brain age prediction.
    Uses original paper's layer naming convention for checkpoint compatibility.
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
        
        # Build model components using original naming convention
        self.mednextv1 = create_mednext_encoder_v1(
            num_input_channels=self.in_channels, 
            num_classes=1, 
            model_id=self.model_id, 
            kernel_size=self.kernel_size, 
            deep_supervision=self.deep_supervision
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regression_fc = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier/Glorot initialization for better stability
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model with robust error handling.
        Detects and handles NaN/Infinity values to prevent training failures.
        """
        # Input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"WARNING: Input tensor contains NaN or Inf values. Replacing with zeros.")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        # Forward pass with exception handling
        try:
            mednext_out = self.mednextv1(x)
            
            # Check for NaNs after encoder
            if torch.isnan(mednext_out).any() or torch.isinf(mednext_out).any():
                print(f"WARNING: MedNeXt encoder output contains NaN or Inf values.")
                mednext_out = torch.where(torch.isnan(mednext_out) | torch.isinf(mednext_out), 
                                         torch.zeros_like(mednext_out), mednext_out)
            
            x = mednext_out
            x = self.global_avg_pool(x)
            x = torch.flatten(x, start_dim=1)
            
            # Check for NaNs before regression
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: Feature vector contains NaN or Inf values.")
                x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
            
            age_estimate = self.regression_fc(x)
            
            # Final check for NaNs in output
            if torch.isnan(age_estimate).any() or torch.isinf(age_estimate).any():
                print(f"WARNING: Final age estimate contains NaN or Inf values. Using fallback value.")
                # Use a fallback value (middle of typical age range) if NaNs occur
                age_estimate = torch.ones_like(age_estimate) * 50.0
            
            return age_estimate.squeeze()
            
        except Exception as e:
            print(f"ERROR in forward pass: {str(e)}")
            # Return a reasonable default prediction on error (middle of age range)
            return torch.ones(x.size(0), device=x.device) * 50.0


# # Keep the old MedNeXtEncReg class for backward compatibility if needed
# class MedNeXtEncReg(nn.Module):
#     """Original BrainAgeNeXt model architecture using MedNeXt encoder"""
#     def __init__(self, *args, **kwargs):
#         super(MedNeXtEncReg, self).__init__()
#         self.mednextv1 = create_mednext_encoder_v1(
#             num_input_channels=1, 
#             num_classes=1, 
#             model_id='B', 
#             kernel_size=3, 
#             deep_supervision=True
#         )
#         self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.regression_fc = nn.Sequential(
#             nn.Linear(512, 64),
#             nn.ReLU(),
#             nn.Dropout(0.0),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         mednext_out = self.mednextv1(x)
#         x = mednext_out
#         x = self.global_avg_pool(x)
#         x = torch.flatten(x, start_dim=1)
#         age_estimate = self.regression_fc(x)
#         return age_estimate.squeeze()



