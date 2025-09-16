from typing import Dict, Tuple, Optional, Union, List

import numpy as np
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    SqueezeDimd, 
    AsDiscreted,
    RandAffined,
    RandFlipd,
    RandRotated,
    RandZoomd
)
from brain_age_pred.dataset.custom_transformations import ConvertLabelsD
from brain_age_pred.brain_gen.labels import GENERATION_LABELS, GENERATION_CLASSES

class SegmentationAugmentationConfig:
    """Configuration for segmentation augmentation transforms."""
    
    # Default probabilities for spatial transforms
    DEFAULT_PROBS = {
        "flip": 0.5,
        "affine": 0.5,
        "zoom": 0.5,
        "rotate": 0.5,
    }
    
    # Default parameters for spatial transforms
    DEFAULT_PARAMS = {
        # spatial ranges
        "scaling_range": (0.95, 1.05),
        "shearing_bounds": 0.2,
        # zoom parameters
        "zoom_min": 0.95,
        "zoom_max": 1.05,
        # rotation parameters
        "rotate_range_x": 0.1,
        "rotate_range_y": 0.1,
        "rotate_range_z": 0.1,
    }
    
    def __init__(
        self,
        transform_probs: Optional[Dict[str, float]] = None,
        scaling_range: Optional[Union[Tuple[float, float], List[float]]] = None,
        shearing_bounds: Optional[float] = None,
        zoom_min: Optional[float] = None,
        zoom_max: Optional[float] = None,
        rotate_range_x: Optional[float] = None,
        rotate_range_y: Optional[float] = None,
        rotate_range_z: Optional[float] = None,
    ):
        # Initialize probabilities with defaults and override with provided values
        self.probs = {**self.DEFAULT_PROBS}
        if transform_probs:
            for k, v in transform_probs.items():
                if k in self.probs:
                    self.probs[k] = v
        
        # Initialize parameters with defaults and override with provided values
        self.params = {**self.DEFAULT_PARAMS}
        param_mapping = {
            'scaling_range': scaling_range,
            'shearing_bounds': shearing_bounds,
            'zoom_min': zoom_min,
            'zoom_max': zoom_max,
            'rotate_range_x': rotate_range_x,
            'rotate_range_y': rotate_range_y,
            'rotate_range_z': rotate_range_z,
        }
        
        for param_name, param_value in param_mapping.items():
            if param_value is not None:
                self.params[param_name] = param_value

def create_augmented_one_hot_transform(n_classes, config=None):
    """
    Create a transform pipeline that includes spatial augmentations before one-hot encoding.
    
    Args:
        n_classes: Number of classes for one-hot encoding
        config: Optional SegmentationAugmentationConfig object with transform settings
    """
    if config is None:
        config = SegmentationAugmentationConfig()
    
    spatial_transforms = [
        # Ensure channel first - data already has channel dimension at position 0
        EnsureChannelFirstd(keys=["image"], channel_dim=0),  # Changed from "no_channel"
        
        # Add spatial augmentations before label conversion
        RandFlipd(
            keys=["image"],
            prob=config.probs["flip"],
            spatial_axis=0,
        ),
        
        RandZoomd(
            keys=["image"], 
            min_zoom=config.params["zoom_min"], 
            max_zoom=config.params["zoom_max"], 
            prob=config.probs["zoom"],
            mode="nearest"
        ),
        
        RandRotated(
            keys=["image"], 
            range_x=config.params["rotate_range_x"], 
            range_y=config.params["rotate_range_y"], 
            range_z=config.params["rotate_range_z"], 
            prob=config.probs["rotate"],
            mode="nearest"
        ),
        
        RandAffined(
            keys=["image"],
            prob=config.probs["affine"],
            rotate_range=(0, 0, 0),  # Disable rotation since we use RandRotated
            scale_range=config.params["scaling_range"],  
            shear_range=(config.params["shearing_bounds"],) * 3,
            mode="nearest",
            padding_mode="zeros"
        ),
        
        # Convert labels and create one-hot encoding
        ConvertLabelsD(
            keys=["image"],
            generation_labels=GENERATION_LABELS,
            output_labels=GENERATION_CLASSES
        ),
        SqueezeDimd(keys=["image"], dim=0),
        AsDiscreted(keys=["image"], to_onehot=n_classes),
    ]
    
    return Compose(spatial_transforms)

# For backwards compatibility, keep the original one_hot_transform
def get_one_hot_transform(n_classes):
    """Get the original one-hot transform without spatial augmentations"""
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim=0),  # Changed from "no_channel"
        ConvertLabelsD(
            keys=["image"],
            generation_labels=GENERATION_LABELS,
            output_labels=GENERATION_CLASSES
        ),
        SqueezeDimd(keys=["image"], dim=0),
        AsDiscreted(keys=["image"], to_onehot=n_classes),
    ])