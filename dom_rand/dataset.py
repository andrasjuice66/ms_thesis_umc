import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from typing import List, Dict, Tuple, Optional, Union, Callable
import random
from .domain_randomization import BrainAgeDomainRandomizer


class BrainMRIDataset(Dataset):
    """
    Optimized dataset for brain MRI images with efficient I/O, caching, and online augmentation.
    Works with the BrainAgeDomainRandomizer for real-time data augmentation.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        age_labels: Optional[List[float]] = None,
        transform: Optional[BrainAgeDomainRandomizer] = None,
        cache_mode: str = "memory",  # Options: "memory", "disk", "none"
        cache_dir: Optional[str] = None,
        target_shape: Optional[Tuple[int, int, int]] = (176, 240, 256),
        preload: bool = True,
        normalize: bool = True,
        device: Optional[torch.device] = None,
        mode: str = "train"  # Options: "train", "val", "test"
    ):
        """
        Initialize the optimized brain MRI dataset.
        
        Args:
            file_paths: List of paths to NIfTI MRI files
            age_labels: Age labels for each file (if None, dataset is for inference only)
            transform: Domain randomization transformer to apply (only used in train mode)
            cache_mode: How to cache data ("memory", "disk", or "none")
            cache_dir: Directory for disk caching (required if cache_mode="disk")
            target_shape: Common shape to resize all images to (needed for batching)
            preload: Whether to preload all data during initialization
            normalize: Whether to normalize image intensities
            device: Device to store tensors on (useful for GPU preprocessing)
            mode: Dataset mode ("train", "val", or "test") - controls whether to apply randomization
        """
        self.file_paths = file_paths
        self.age_labels = age_labels
        self.transform = transform
        self.cache_mode = cache_mode
        self.cache_dir = cache_dir
        self.target_shape = target_shape
        self.normalize = normalize
        self.device = device if device is not None else torch.device("cpu")
        self.mode = mode.lower()
        
        # Validate inputs
        if self.cache_mode == "disk" and self.cache_dir is None:
            raise ValueError("cache_dir must be provided when cache_mode='disk'")
        
        if self.cache_mode == "disk":
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Initialize cache
        self.cache = {}
        
        # Metadata for the MRI files (will be populated during loading)
        self.metadata = []
        
        # Preload data if requested
        if preload:
            self._preload_data()
            
    def _preload_data(self):
        """Preload all MRI files to accelerate training."""
        print(f"Preloading {len(self.file_paths)} MRI files...")
        
        for idx, file_path in enumerate(self.file_paths):
            file_name = os.path.basename(file_path)
            print(f"Loading {idx+1}/{len(self.file_paths)}: {file_name}")
            
            # Load and preprocess image
            self._load_and_cache(idx)
            
        print("Preloading complete!")
    
    def _load_and_cache(self, idx):
        """Load a single MRI file and store it in the cache."""
        file_path = self.file_paths[idx]
        file_name = os.path.basename(file_path)
        
        # Check if already cached
        if idx in self.cache:
            return self.cache[idx]
            
        # Read the NIfTI file
        nii_img = nib.load(file_path)
        
        # Extract metadata and store it
        metadata = {
            "affine": nii_img.affine,
            "header": nii_img.header,
            "shape": nii_img.shape,
            "file_name": file_name,
        }
        
        if len(self.metadata) <= idx:
            self.metadata.append(metadata)
        else:
            self.metadata[idx] = metadata
            
        # Get image data as numpy array
        img_data = nii_img.get_fdata().astype(np.float32)
        
        # Normalize if requested
        if self.normalize:
            img_data = self._normalize_intensity(img_data)
            
        # Resize to target shape if provided
        if self.target_shape:
            # Convert to TorchIO and resize
            tensor_data = torch.from_numpy(img_data).unsqueeze(0)  # Add channel dim
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=tensor_data)
            )
            resizer = tio.Resize(self.target_shape)
            subject = resizer(subject)
            img_data = subject.image.data.squeeze(0).numpy()  # Remove channel dim
        
        # Store in memory cache if requested
        if self.cache_mode == "memory":
            self.cache[idx] = img_data
            
        # Store in disk cache if requested
        elif self.cache_mode == "disk":
            cache_path = os.path.join(self.cache_dir, f"{idx}.npy")
            np.save(cache_path, img_data)
            
        return img_data
    
    def _normalize_intensity(self, img_data):
        """Normalize image intensities to [0, 1] range."""
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        
        if max_val > min_val:
            return (img_data - min_val) / (max_val - min_val)
        return img_data  # Return unchanged if degenerate
        
    def __len__(self):
        """Get the dataset size."""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Get a single preprocessed MRI sample with augmentation."""
        # Load image data (from cache or file)
        if self.cache_mode == "memory" and idx in self.cache:
            img_data = self.cache[idx]
        elif self.cache_mode == "disk":
            cache_path = os.path.join(self.cache_dir, f"{idx}.npy")
            if os.path.exists(cache_path):
                img_data = np.load(cache_path)
            else:
                img_data = self._load_and_cache(idx)
        else:
            img_data = self._load_and_cache(idx)
            
        # Convert to tensor
        img_tensor = torch.from_numpy(img_data).unsqueeze(0)  # Add channel dimension
        
        # Prepare sample dictionary
        sample_dict = {"image": img_tensor}
        
        # Add age label if available
        if self.age_labels is not None:
            sample_dict["age"] = torch.tensor(self.age_labels[idx], dtype=torch.float32)
            
        # Apply domain randomization ONLY if in training mode
        if self.transform is not None and self.mode == "train":
            sample_dict = self.transform(sample_dict)
            
        return sample_dict
    
    def create_dataloader(
        self, 
        batch_size=8, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=None
    ):
        """
        Create an optimized DataLoader for this dataset.
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Keep worker processes alive between epochs
            prefetch_factor: Number of batches to prefetch per worker
            worker_init_fn: Function to initialize workers
            
        Returns:
            An optimized DataLoader instance
        """
        # Define a worker initialization function for better randomness
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        # Use provided worker_init_fn or the default one
        worker_init = worker_init_fn if worker_init_fn else seed_worker
        
        # Create and return the dataloader
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            worker_init_fn=worker_init,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    @classmethod
    def from_directory(
        cls,
        directory_path: str,
        age_file: Optional[str] = None,
        transform: Optional[BrainAgeDomainRandomizer] = None,
        extension: str = ".nii.gz",
        mode: str = "train",
        **kwargs
    ):
        """
        Create a dataset from a directory of MRI files.
        
        Args:
            directory_path: Path to directory containing MRI files
            age_file: Path to CSV file with age labels (format: filename,age)
            transform: Domain randomization transformer to apply
            extension: File extension to look for
            mode: Dataset mode ("train", "val", or "test")
            **kwargs: Additional arguments to pass to constructor
            
        Returns:
            A BrainMRIDataset instance
        """
        # Collect all files with the specified extension
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(extension):
                    file_paths.append(os.path.join(root, file))
                    
        # Sort for reproducibility
        file_paths.sort()
        
        # Load age labels if provided
        age_labels = None
        if age_file:
            age_dict = {}
            with open(age_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            filename = parts[0]
                            age = float(parts[1])
                            age_dict[filename] = age
            
            # Match age labels to file paths
            age_labels = []
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                if file_name in age_dict:
                    age_labels.append(age_dict[file_name])
                else:
                    raise ValueError(f"No age label found for {file_name}")
        
        # Create and return dataset
        return cls(
            file_paths=file_paths,
            age_labels=age_labels,
            transform=transform,
            mode=mode,
            **kwargs
        )
