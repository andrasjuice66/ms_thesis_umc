import torch
import numpy as np
import monai
from monai.transforms import (
    Compose, 
    RandAffined, 
    RandBiasFieldd, 
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    ToTensord
)
import torchio as tio
from typing import Dict, List, Optional, Tuple, Union
import random
from torch.utils.data import DataLoader


class BrainAgeDomainRandomizer:
    """
    Domain randomization for brain MRI images to create diverse datasets for age prediction.
    Implements efficient online augmentation with configurable parameters.
    """
    
    def __init__(
        self,
        image_key: str = "image",
        age_key: str = "age",
        
        # Intensity parameters
        intensity_bounds: Tuple[float, float] = (0, 100),
        contrast_bounds: Tuple[float, float] = (0.75, 1.25),
        
        # Spatial deformation parameters
        flipping: bool = True,
        scaling_bounds: float = 0.2,
        rotation_bounds: float = 15,
        shearing_bounds: float = 0.012,
        nonlin_std: float = 3.0,
        
        # Blurring/resampling parameters
        randomise_res: bool = True,
        max_res_iso: float = 3.0,
        max_res_aniso: float = 5.0,
        
        # Bias field parameters
        bias_field_std: float = 0.5,
        
        # Progressive augmentation parameters
        progressive_mode: bool = False,
        current_epoch: int = 0,
        max_epochs: int = 100,
        
        # Performance parameters
        device: torch.device = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        use_cache: bool = True,
        cache_size: int = 100,
    ):
        """
        Initialize the brain age domain randomizer with configurable parameters.
        """
        self.image_key = image_key
        self.age_key = age_key
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add a flag to track if we're using GPU
        self.use_gpu = self.device.type == "cuda"
        
        # Augmentation parameters
        self.intensity_bounds = intensity_bounds
        self.contrast_bounds = contrast_bounds
        self.flipping = flipping
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.nonlin_std = nonlin_std
        self.randomise_res = randomise_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        self.bias_field_std = bias_field_std
        
        # Progressive augmentation parameters
        self.progressive_mode = progressive_mode
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        
        # Performance parameters
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.cache = {}
        
        # Initialize transforms
        self._initialize_transforms()
    
    def _initialize_transforms(self):
        """Initialize all transformation pipelines."""
        # Create separate transform pipelines for different types of augmentations
        self.spatial_transforms = self._get_spatial_transforms()
        self.intensity_transforms = self._get_intensity_transforms()
        self.resolution_transforms = self._get_resolution_transforms()
        self.artifact_transforms = self._get_artifact_transforms()
        
        # Create TorchIO transforms for more advanced operations
        self.torchio_transforms = self._get_torchio_transforms()
        
        # Final transform to ensure correct output format
        self.final_transform = Compose([
            ToTensord(keys=[self.image_key])
        ])
        
        # Set device for MONAI transforms that support it
        if self.use_gpu:
            for transform in [self.spatial_transforms["monai"], 
                             self.intensity_transforms["monai"]]:
                for t in transform.transforms:
                    if hasattr(t, "set_device"):
                        t.set_device(self.device)
    
    def _get_spatial_transforms(self):
        """Create spatial transformations (affine, elastic deformations)."""
        # Process parameters to match expected format
        if isinstance(self.scaling_bounds, (int, float)):
            scale_range = (1 - self.scaling_bounds, 1 + self.scaling_bounds)
        else:
            scale_range = self.scaling_bounds
            
        # Create MONAI transforms
        monai_transforms = Compose([
            RandAffined(
                keys=[self.image_key],
                prob=0.8,
                rotate_range=[self.rotation_bounds * np.pi/180] * 3,
                shear_range=[self.shearing_bounds] * 3,
                scale_range=[scale_range] * 3,
                padding_mode="zeros",
                mode="bilinear"
            )
        ])
        
        # Create TorchIO transforms for elastic deformations
        torchio_transforms = tio.Compose([
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=self.nonlin_std,
                locked_borders=2
            )
        ])
        
        return {"monai": monai_transforms, "torchio": torchio_transforms}
    
    def _get_intensity_transforms(self):
        """Create intensity transformations."""
        monai_transforms = Compose([
            RandAdjustContrastd(
                keys=[self.image_key],
                prob=0.7,
                gamma=self.contrast_bounds
            ),
            RandHistogramShiftd(
                keys=[self.image_key],
                prob=0.6,
                num_control_points=10
            )
        ])
        
        torchio_transforms = tio.Compose([
            tio.RandomGamma(
                log_gamma=(-0.3, 0.3),
                p=0.7
            ),
            tio.RandomNoise(
                mean=0,
                std=(0, 0.1),
                p=0.5
            )
        ])
        
        return {"monai": monai_transforms, "torchio": torchio_transforms}
    
    def _get_resolution_transforms(self):
        """Create resolution transformations (blurring, downsampling)."""
        monai_transforms = Compose([
            RandGaussianSmoothd(
                keys=[self.image_key],
                prob=0.5 if self.randomise_res else 0.0,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5)
            )
        ])
        
        torchio_transforms = tio.Compose([
            tio.RandomAnisotropy(
                axes=(0, 1, 2),
                downsampling=(1, self.max_res_aniso),
                p=0.5 if self.randomise_res else 0.0
            )
        ])
        
        return {"monai": monai_transforms, "torchio": torchio_transforms}
    
    def _get_artifact_transforms(self):
        """Create artifact transformations (bias field, noise)."""
        monai_transforms = Compose([
            RandBiasFieldd(
                keys=[self.image_key],
                prob=0.7,
                coeff_range=(0.0, self.bias_field_std)
            ),
            RandGaussianNoised(
                keys=[self.image_key],
                prob=0.5,
                mean=0.0,
                std=0.1
            )
        ])
        
        torchio_transforms = tio.Compose([
            tio.RandomGhosting(
                num_ghosts=(2, 10),
                axes=(0, 1, 2),
                p=0.3
            ),
            tio.RandomSpike(
                num_spikes=(1, 5),
                intensity=(0.1, 0.5),
                p=0.2
            )
        ])
        
        return {"monai": monai_transforms, "torchio": torchio_transforms}
    
    def _get_torchio_transforms(self):
        """Create combined TorchIO transforms."""
        return tio.Compose([
            tio.OneOf({
                tio.RandomMotion(
                    degrees=self.rotation_bounds,
                    translation=self.scaling_bounds,
                    num_transforms=3,
                    p=0.3
                ): 0.3,
                tio.RandomGhosting(
                    num_ghosts=(2, 10),
                    axes=(0, 1, 2),
                    p=0.3
                ): 0.3,
                tio.RandomBiasField(
                    coefficients=self.bias_field_std,
                    p=0.3
                ): 0.4
            })
        ])
    
    def _convert_to_torchio_subject(self, data_dict):
        """Convert a data dictionary to a TorchIO subject."""
        subject_dict = {
            self.image_key: tio.ScalarImage(tensor=data_dict[self.image_key])
        }
        return tio.Subject(subject_dict)
    
    def _convert_from_torchio_subject(self, subject, data_dict):
        """Convert a TorchIO subject back to a data dictionary."""
        data_dict[self.image_key] = subject[self.image_key].data
        return data_dict
    
    def get_augmentation_intensity(self):
        """Get the current augmentation intensity based on training progress."""
        if not self.progressive_mode:
            return 1.0
        
        # Gradually increase augmentation intensity
        return min(1.0, self.current_epoch / (self.max_epochs * 0.7))
    
    def __call__(self, data_dict):
        """Apply domain randomization to a data dictionary."""
        # Check cache if enabled
        if self.use_cache:
            cache_key = hash(data_dict[self.image_key].tobytes())
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Get augmentation intensity for progressive augmentation
        aug_intensity = self.get_augmentation_intensity()
        
        # Skip augmentation with some probability based on intensity
        if self.progressive_mode and random.random() > aug_intensity:
            result = self.final_transform(data_dict)
            return result
        
        # Convert to TorchIO subject for some transforms
        subject = self._convert_to_torchio_subject(data_dict)
        
        # Apply spatial transforms (TorchIO)
        if random.random() < aug_intensity:
            subject = self.spatial_transforms["torchio"](subject)
        
        # Apply resolution transforms (TorchIO)
        if random.random() < aug_intensity * 0.8:
            subject = self.resolution_transforms["torchio"](subject)
        
        # Apply artifact transforms (TorchIO)
        if random.random() < aug_intensity * 0.6:
            subject = self.artifact_transforms["torchio"](subject)
        
        # Convert back to dictionary for MONAI transforms
        data_dict = self._convert_from_torchio_subject(subject, data_dict)
        
        # Apply spatial transforms (MONAI)
        if random.random() < aug_intensity:
            data_dict = self.spatial_transforms["monai"](data_dict)
        
        # Apply intensity transforms (MONAI)
        if random.random() < aug_intensity * 0.9:
            data_dict = self.intensity_transforms["monai"](data_dict)
        
        # Apply resolution transforms (MONAI)
        if random.random() < aug_intensity * 0.7:
            data_dict = self.resolution_transforms["monai"](data_dict)
        
        # Apply artifact transforms (MONAI)
        if random.random() < aug_intensity * 0.5:
            data_dict = self.artifact_transforms["monai"](data_dict)
        
        # Apply final transforms
        data_dict = self.final_transform(data_dict)
        
        # Cache result if enabled
        if self.use_cache:
            if len(self.cache) >= self.cache_size:
                # Remove a random key to keep cache size in check
                self.cache.pop(random.choice(list(self.cache.keys())))
            self.cache[cache_key] = data_dict
        
        return data_dict 

    def create_dataloader(self, dataset, batch_size=8, shuffle=True):
        """
        Create an optimized DataLoader with this domain randomizer.
        
        Args:
            dataset: The dataset to load from
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            A configured DataLoader
        """
        # Use persistent workers to avoid initialization overhead
        persistent_workers = self.num_workers > 0
        
        # Pin memory for faster CPU->GPU transfer
        pin_memory = self.use_gpu
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        ) 