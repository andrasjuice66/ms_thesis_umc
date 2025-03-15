"""
Dataset classes and utilities for brain age prediction.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.transforms import (
    Compose, 
    LoadImaged, 
    AddChanneld, 
    ScaleIntensityd, 
    Orientationd, 
    Spacingd,
    ToTensord
)
from monai.data import CacheDataset, PersistentDataset
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path

from dom_rand.domain_randomization import DomainRandomizer


class BrainAgeDataset(Dataset):
    """
    Dataset for brain age prediction with domain randomization.
    """
    
    def __init__(
        self,
        data_csv: str,
        image_dir: str,
        image_key: str = "image",
        age_key: str = "age",
        transform=None,
        domain_randomizer=None,
        cache_dir: Optional[str] = None,
        mode: str = "train",
        preload: bool = False,
        num_workers: int = 4
    ):
        """
        Initialize the brain age dataset.
        
        Args:
            data_csv: Path to CSV file with image paths and ages
            image_dir: Directory containing the images
            image_key: Key for the image in the data dictionary
            age_key: Key for the age in the data dictionary
            transform: MONAI transforms for preprocessing
            domain_randomizer: Domain randomization module
            cache_dir: Directory to cache preprocessed data
            mode: 'train', 'val', or 'test'
            preload: Whether to preload all data into memory
            num_workers: Number of workers for data loading
        """
        self.data_csv = data_csv
        self.image_dir = Path(image_dir)
        self.image_key = image_key
        self.age_key = age_key
        self.transform = transform
        self.domain_randomizer = domain_randomizer
        self.cache_dir = cache_dir
        self.mode = mode
        self.preload = preload
        self.num_workers = num_workers
        
        # Load data CSV
        self.data_df = pd.read_csv(data_csv)
        
        # Create default transform if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()
        
        # Preload data if requested
        self.cache = {}
        if self.preload:
            self._preload_data()
    
    def _get_default_transform(self):
        """Get default transform for preprocessing."""
        return Compose([
            LoadImaged(keys=[self.image_key]),
            AddChanneld(keys=[self.image_key]),
            Orientationd(keys=[self.image_key], axcodes="RAS"),
            Spacingd(keys=[self.image_key], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityd(keys=[self.image_key], minv=0.0, maxv=1.0),
            ToTensord(keys=[self.image_key])
        ])
    
    def _preload_data(self):
        """Preload all data into memory."""
        for i in range(len(self.data_df)):
            self.cache[i] = self.__getitem__(i, use_cache=False)
    
    def __len__(self):
        """Get dataset length."""
        return len(self.data_df)
    
    def __getitem__(self, idx, use_cache=True):
        """
        Get dataset item.
        
        Args:
            idx: Index
            use_cache: Whether to use cache
            
        Returns:
            Data dictionary
        """
        # Return from cache if available
        if use_cache and self.preload and idx in self.cache:
            return self.cache[idx]
        
        # Get image path and age
        row = self.data_df.iloc[idx]
        image_path = self.image_dir / row["image_path"]
        age = row[self.age_key]
        
        # Create data dictionary
        data_dict = {
            self.image_key: str(image_path),
            self.age_key: age
        }
        
        # Apply transform
        if self.transform:
            data_dict = self.transform(data_dict)
        
        # Apply domain randomization for training
        if self.mode == "train" and self.domain_randomizer is not None:
            data_dict = self.domain_randomizer(data_dict)
        
        return data_dict


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    image_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    domain_randomization_params: Optional[Dict] = None,
    progressive_mode: bool = False,
    current_epoch: int = 0,
    max_epochs: int = 100
):
    """
    Create data loaders for training and validation.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        image_dir: Directory containing the images
        batch_size: Batch size
        num_workers: Number of workers for data loading
        cache_dir: Directory to cache preprocessed data
        domain_randomization_params: Parameters for domain randomization
        progressive_mode: Whether to use progressive domain randomization
        current_epoch: Current epoch for progressive mode
        max_epochs: Maximum number of epochs for progressive mode
        
    Returns:
        Training and validation data loaders
    """
    # Create domain randomizer
    if domain_randomization_params is None:
        domain_randomization_params = {}
    
    domain_randomizer = DomainRandomizer(
        augmentation_strength=domain_randomization_params.get("augmentation_strength", "medium"),
        **domain_randomization_params
    )
    
    # Create datasets
    train_dataset = BrainAgeDataset(
        data_csv=train_csv,
        image_dir=image_dir,
        domain_randomizer=domain_randomizer,
        cache_dir=cache_dir,
        mode="train",
        num_workers=num_workers
    )
    
    val_dataset = BrainAgeDataset(
        data_csv=val_csv,
        image_dir=image_dir,
        domain_randomizer=None,  # No randomization for validation
        cache_dir=cache_dir,
        mode="val",
        num_workers=num_workers
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 