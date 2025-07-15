from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import torch
import nibabel as nib
from monai.transforms import (
    Compose, ToTensord, CopyItemsd, DeleteItemsd, Transform,
    CenterSpatialCropd
)

from brain_age_pred.dataset.custom_transformations import (
    ConvertLabelsD, IntensityClipNormalizeD
)
from brain_age_pred.brain_gen.labels import (
    GENERATION_LABELS, GENERATION_CLASSES
)


class ValidationGenerator:
    """
    Simple generator for validation/test that loads real images 
    and their corresponding segmentations from separate directories.
    
    Unlike BABrainGenerator, this doesn't do any synthetic generation
    or augmentation - just loads real data and formats it properly.
    """
    
    def __init__(
        self,
        segmented_data_dir: str | Path,
        generation_labels: np.ndarray | None = None,
        output_labels: np.ndarray | None = None,
        use_intensity_clip_normalize: bool = True,
        return_segmentation: bool = True,
        output_shape: tuple[int, int, int] = (160, 192, 160),
    ):
        self.segmented_data_dir = Path(segmented_data_dir)
        self.image_key = "image"
        
        # Label arrays
        self.generation_labels = generation_labels if generation_labels is not None else GENERATION_LABELS
        self.output_labels = output_labels if output_labels is not None else self.generation_labels
        
        # Settings
        self.use_intensity_clip_normalize = use_intensity_clip_normalize
        self.return_segmentation = return_segmentation
        self.output_shape = output_shape
        
        self._build_pipeline()
    
    def _build_pipeline(self):
        tx = []
        
        # Always convert labels to ensure consistency
        tx.append(
            ConvertLabelsD(
                keys=["seg_gt"],
                generation_labels=self.generation_labels,
                output_labels=self.output_labels,
                background_label=0
            )
        )
        
        # Optional intensity normalization for the real image
        if self.use_intensity_clip_normalize:
            tx.append(
                IntensityClipNormalizeD(
                    keys=[self.image_key],
                    clip_percentiles=(1.0, 99.0),
                    normalise=True,
                    gamma_std=0.0,  # No gamma augmentation for validation
                    separate_channels=True,
                    prob=1.0  # Always apply for consistency
                )
            )
        
        # Clean-up & tensor conversion
        keys_to_delete = []
        if not self.return_segmentation:
            keys_to_delete.append("seg_gt")
        
        if keys_to_delete:
            tx.append(DeleteItemsd(keys=keys_to_delete))
        
        tensor_keys = [self.image_key]
        if self.return_segmentation:
            tensor_keys.append("seg_gt")
        
        tx.append(ToTensord(keys=tensor_keys, allow_missing_keys=True))
        
        self.transform = Compose(tx)
    
    @staticmethod
    def _load_volume(path: str | Path) -> np.ndarray:
        """Load volume from file"""
        path = str(path)
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith((".nii.gz", ".nii")):
            return nib.load(path).get_fdata()
        else:
            raise ValueError(f"Unsupported file extension: {path}")
    
    def _get_segmentation_path(self, image_path: str | Path) -> Path:
        """
        Get corresponding segmentation path from image path.
        Assumes same relative path structure in segmented_data_dir.
        """
        image_path = Path(image_path)
        
        # Extract relative path (assuming image_path contains the relative part)
        # This assumes your CSV contains relative paths like "dataset/subject/image.nii.gz"
        if "brain_age_preprocessed" in str(image_path):
            # Extract everything after "brain_age_preprocessed/"
            parts = image_path.parts
            preprocessed_idx = None
            for i, part in enumerate(parts):
                if "brain_age_preprocessed" in part:
                    preprocessed_idx = i
                    break
            
            if preprocessed_idx is not None:
                rel_path = Path(*parts[preprocessed_idx + 1:])
                seg_path = self.segmented_data_dir / rel_path
                return seg_path
        
        # Fallback: try to extract filename and search
        filename = image_path.name
        # Search for this filename in segmented_data_dir
        for seg_file in self.segmented_data_dir.rglob(filename):
            return seg_file
        
        raise FileNotFoundError(f"Could not find segmentation for {image_path}")
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process sample: load corresponding segmentation and format both.
        
        Input sample should contain:
        - 'image': real image tensor or path to real image
        - '__image_path__': path to the image file (if image is already loaded)
        """
        
        # Get the image path to find corresponding segmentation
        if isinstance(sample.get(self.image_key), (str, Path)):
            # Image is still a path, load it
            image_path = sample[self.image_key]
            real_img = self._load_volume(image_path)
            sample[self.image_key] = torch.from_numpy(real_img).unsqueeze(0).float()
        else:
            # Image is already loaded, need path from metadata
            image_path = sample.get('__image_path__')
            if image_path is None:
                raise ValueError("Need either image path or __image_path__ in sample")
        
        # Load corresponding segmentation
        try:
            seg_path = self._get_segmentation_path(image_path)
            seg_img = self._load_volume(seg_path)
            sample["seg_gt"] = torch.from_numpy(seg_img).unsqueeze(0).float()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            # Create dummy segmentation if not found
            img_shape = sample[self.image_key].shape[1:]  # Remove channel dim
            sample["seg_gt"] = torch.zeros((1, *img_shape), dtype=torch.float32)
        
        # Apply transformations
        sample = self.transform(sample)
        
        return sample
