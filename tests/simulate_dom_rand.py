"""
Test script for domain randomization, showing which transformations were applied.

This script:
1. Loads configuration from a YAML file
2. Loads a sample 3D brain MRI image
3. Creates a modified DomainRandomizer that tracks which transforms were applied
4. Applies the randomizer multiple times to the same image
5. Logs and displays which transformations were used each time
"""

import os
import sys
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import nibabel as nib
from monai.transforms import Compose, RandAffined, RandAdjustContrastd, RandBiasFieldd, RandFlipd, RandGaussianSmoothd, \
    RandGaussianNoised, RandRicianNoised, RandScaleIntensityd, RandShiftIntensityd, RandHistogramShiftd, RandGibbsNoised, \
    RandCoarseDropoutd, RandSpatialCropd, ToTensord
import torchio as tio

# Add the project root to the path so we can import the module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.dataset.domain_randomization import DomainRandomizer
from brain_age_pred.dataset.custom_transformations import RandomResolutionD, RandGammaD

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrackedTransform:
    """Wrapper around a transform that tracks whether it was applied."""
    def __init__(self, transform, name, prob_key=None):
        self.transform = transform
        self.name = name
        self.was_applied = False
        self.prob_key = prob_key  # The key in the probs dictionary
        
    def __call__(self, data):
        # Store the original data to compare later
        original = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                   for k, v in data.items()}
        
        # Apply the transform
        result = self.transform(data)
        
        # Check if the image data was modified
        image_key = next((k for k in data.keys() if 'image' in k.lower()), None)
        if image_key:
            if not torch.allclose(original[image_key], result[image_key]):
                self.was_applied = True
        
        return result

class TrackingDomainRandomizer(DomainRandomizer):
    """Extension of DomainRandomizer that tracks which transforms were applied."""
    
    def _build_monai_pipeline(self) -> None:
        """Override to add tracking wrappers around transforms."""
        deg2rad = np.pi / 180
        tfms = []
        self.tracked_transforms = {}

        # 1. flips & affine
        flip_transform = RandFlipd(
            keys=[self.image_key],
            prob=self.prob["flip"],
            spatial_axis=0,
        )
        tracked_flip = TrackedTransform(flip_transform, "Flip", "flip")
        tfms.append(tracked_flip)
        self.tracked_transforms["flip"] = tracked_flip
        
        affine_transform = RandAffined(
            keys=[self.image_key],
            prob=self.prob["affine"],
            rotate_range=(deg2rad * self.rotation_range,) * 3,
            scale_range=(self.scaling_range[1] - 1,) * 3,
            shear_range=(self.shearing_bounds,) * 3,
            mode="bilinear",
        )
        tracked_affine = TrackedTransform(affine_transform, "Affine", "affine")
        tfms.append(tracked_affine)
        self.tracked_transforms["affine"] = tracked_affine

        # 2. basic intensity
        contrast_transform = RandAdjustContrastd(
            keys=[self.image_key],
            prob=self.prob["contrast"],
            gamma=self.contrast_range,
        )
        tracked_contrast = TrackedTransform(contrast_transform, "Contrast", "contrast")
        tfms.append(tracked_contrast)
        self.tracked_transforms["contrast"] = tracked_contrast
        
        gamma_transform = RandGammaD(
            keys=[self.image_key],
            log_gamma_std=self.log_gamma_std,
            prob=self.prob["gamma"],
        )
        tracked_gamma = TrackedTransform(gamma_transform, "Gamma", "gamma")
        tfms.append(tracked_gamma)
        self.tracked_transforms["gamma"] = tracked_gamma
        
        scale_int_transform = RandScaleIntensityd(
            keys=[self.image_key],
            prob=self.prob["scale_int"],
            factors=self.contrast_range,
        )
        tracked_scale_int = TrackedTransform(scale_int_transform, "Scale Intensity", "scale_int")
        tfms.append(tracked_scale_int)
        self.tracked_transforms["scale_int"] = tracked_scale_int
        
        shift_int_transform = RandShiftIntensityd(
            keys=[self.image_key],
            prob=self.prob["shift_int"],
            offsets=self.shift_offset,
        )
        tracked_shift_int = TrackedTransform(shift_int_transform, "Shift Intensity", "shift_int")
        tfms.append(tracked_shift_int)
        self.tracked_transforms["shift_int"] = tracked_shift_int
        
        hist_shift_transform = RandHistogramShiftd(
            keys=[self.image_key],
            prob=self.prob["hist_shift"],
            num_control_points=self.hist_control_points,
        )
        tracked_hist_shift = TrackedTransform(hist_shift_transform, "Histogram Shift", "hist_shift")
        tfms.append(tracked_hist_shift)
        self.tracked_transforms["hist_shift"] = tracked_hist_shift

        # 3. noise / artefacts
        noise_transform = RandGaussianNoised(
            keys=[self.image_key],
            prob=self.prob["noise"],
            mean=self.noise_mean,
            std=self.noise_std,
        )
        tracked_noise = TrackedTransform(noise_transform, "Gaussian Noise", "noise")
        tfms.append(tracked_noise)
        self.tracked_transforms["noise"] = tracked_noise
        
        rician_transform = RandRicianNoised(
            keys=[self.image_key],
            prob=self.prob["rician"],
            std=self.rician_std,
        )
        tracked_rician = TrackedTransform(rician_transform, "Rician Noise", "rician")
        tfms.append(tracked_rician)
        self.tracked_transforms["rician"] = tracked_rician
        
        gibbs_transform = RandGibbsNoised(
            keys=[self.image_key],
            prob=self.prob["gibbs"],
            alpha=self.gibbs_alpha,
        )
        tracked_gibbs = TrackedTransform(gibbs_transform, "Gibbs Noise", "gibbs")
        tfms.append(tracked_gibbs)
        self.tracked_transforms["gibbs"] = tracked_gibbs
        
        blur_transform = RandGaussianSmoothd(
            keys=[self.image_key],
            prob=self.prob["blur"],
            sigma_x=self.blur_sigma,
            sigma_y=self.blur_sigma,
            sigma_z=self.blur_sigma,
        )
        tracked_blur = TrackedTransform(blur_transform, "Blur", "blur")
        tfms.append(tracked_blur)
        self.tracked_transforms["blur"] = tracked_blur
        
        bias_transform = RandBiasFieldd(
            keys=[self.image_key],
            prob=self.prob["bias"],
            coeff_range=self.bias_field_rng,
        )
        tracked_bias = TrackedTransform(bias_transform, "Bias Field", "bias")
        tfms.append(tracked_bias)
        self.tracked_transforms["bias"] = tracked_bias
        
        resolution_transform = RandomResolutionD(
            keys=[self.image_key],
            min_res=self.min_res,
            max_res_iso=self.max_res_iso,
            prob=self.prob["resolution"],
        )
        tracked_resolution = TrackedTransform(resolution_transform, "Resolution", "resolution")
        tfms.append(tracked_resolution)
        self.tracked_transforms["resolution"] = tracked_resolution
        
        coarse_do_transform = RandCoarseDropoutd(
            keys=[self.image_key],
            prob=self.prob["coarse_do"],
            holes=self.coarse_holes,
            spatial_size=self.coarse_size,
            fill_value=0.0,
        )
        tracked_coarse_do = TrackedTransform(coarse_do_transform, "Coarse Dropout", "coarse_do")
        tfms.append(tracked_coarse_do)
        self.tracked_transforms["coarse_do"] = tracked_coarse_do

        # 4. optional crop to ROI
        if self.output_shape is not None:
            crop_transform = RandSpatialCropd(
                keys=[self.image_key],
                roi_size=self.output_shape,
                random_center=self.random_center,
                random_size=False,
            )
            tracked_crop = TrackedTransform(crop_transform, "Spatial Crop", "crop")
            tfms.append(tracked_crop)
            self.tracked_transforms["crop"] = tracked_crop

        # 5. tensor conversion
        tfms.append(ToTensord(keys=[self.image_key]))

        # compose & push to GPU if possible
        self.monai = Compose(tfms)
        if self.device.type == "cuda":
            for t in self.monai.transforms:
                if hasattr(t, "transform") and hasattr(t.transform, "set_device"):
                    t.transform.set_device(self.device)
    
    def _build_torchio_pipeline(self) -> None:
        """Override to add tracking wrappers around TorchIO transforms."""
        if not self.use_tio:
            self.tio = None
            self.torchio_tracked_transforms = {}
            return

        self.torchio_tracked_transforms = {}
        
        # Create the TorchIO transforms
        elastic_transform = tio.RandomElasticDeformation(
            num_control_points=self.elastic_control_points,
            max_displacement=self.elastic_max_displacement,
            locked_borders=2,
            p=self.prob["elastic"],
        )
        
        spike_transform = tio.RandomSpike(
            num_spikes=self.spike_num,
            intensity=self.spike_intensity,
            p=self.prob["spike"],
        )
        
        ghost_transform = tio.RandomGhosting(
            num_ghosts=self.ghost_num,
            axes=(0, 1, 2),
            p=self.prob["ghost"],
        )
        
        # We need a special way to track TorchIO transforms since they work differently
        self.torchio_tracked_transforms = {
            "elastic": {"transform": elastic_transform, "was_applied": False},
            "spike": {"transform": spike_transform, "was_applied": False},
            "ghost": {"transform": ghost_transform, "was_applied": False},
        }
        
        # Create the TorchIO pipeline
        self.tio = tio.Compose([
            elastic_transform,
            spike_transform,
            ghost_transform,
        ])
    
    def __call__(self, sample: dict) -> dict:
        """Apply transformations and track which ones were applied."""
        # Reset tracking for all transforms
        for transform in self.tracked_transforms.values():
            transform.was_applied = False
        
        for transform_info in self.torchio_tracked_transforms.values():
            transform_info["was_applied"] = False
        
        # Process with TorchIO if enabled
        img = sample[self.image_key]
        if self.tio is not None:
            # Store the original image for comparison
            original_img = img.clone()
            
            # Apply TorchIO transforms
            subj = tio.Subject({self.image_key: tio.ScalarImage(tensor=img)})
            result = self.tio(subj)
            img = result[self.image_key].data
            
            # Check which TorchIO transforms were applied by looking at data changes
            # This is approximate since multiple transforms might have changed the data
            if not torch.allclose(original_img, img):
                # At least one transform was applied - we can check individual histories
                for name, info in self.torchio_tracked_transforms.items():
                    transform = info["transform"]
                    # Access the last_history property to check if it was applied
                    if hasattr(transform, 'last_history') and transform.last_history:
                        info["was_applied"] = True
        
        # Apply MONAI transforms (which are already tracked)
        transform_input = {self.image_key: img}
        result = self.monai(transform_input)
        
        # Update the sample with the transformed image
        if result is None:
            raise RuntimeError("DomainRandomizer: MONAI pipeline returned None")
        
        if self.image_key not in result:
            raise RuntimeError(f"DomainRandomizer: Image key '{self.image_key}' missing after transforms")
        
        img = result[self.image_key]
        if img is None:
            raise RuntimeError(f"DomainRandomizer: Image is None after transforms")
        
        # Keep tensors on the same device
        sample[self.image_key] = img
        for k in ("age", "weight"):
            if k in sample and sample[k].device != img.device:
                sample[k] = sample[k].to(img.device)
        
        return sample
    
    def get_applied_transforms(self):
        """Return a dictionary of transforms that were applied."""
        applied = {}
        for name, transform in self.tracked_transforms.items():
            applied[name] = transform.was_applied
        
        for name, info in self.torchio_tracked_transforms.items():
            applied[name] = info["was_applied"]
        
        return applied

def load_sample_image(path):
    """
    Load a sample image for testing, supporting both NIfTI and NumPy formats.
    This simulates how images are loaded in the actual training pipeline.
    """
    try:
        if path is None:
            raise ValueError("No path provided")
        
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            # Load NIfTI file (what's used in actual training)
            logger.info(f"Loading NIfTI file: {path}")
            nii = nib.load(path)
            data = nii.get_fdata()
            
            # Convert to float32 (as done in actual dataset)
            data = data.astype(np.float32)
            
            # Convert to torch tensor with channel dimension (C,D,H,W)
            img = torch.from_numpy(data).float().unsqueeze(0)
            logger.info(f"Loaded image with shape: {img.shape}")
            return img
            
        elif path.endswith('.npy'):
            # Load NumPy file (alternative format)
            logger.info(f"Loading NumPy file: {path}")
            data = np.load(path)
            img = torch.from_numpy(data).float().unsqueeze(0)
            logger.info(f"Loaded image with shape: {img.shape}")
            return img
            
        else:
            raise ValueError(f"Unsupported file format: {path}")
            
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        # Create a synthetic test image if loading fails
        logger.info("Creating synthetic test image")
        # Create a 3D gaussian blob
        size = 128
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, size),
            np.linspace(-1, 1, size),
            np.linspace(-1, 1, size)
        )
        d = np.sqrt(x*x + y*y + z*z)
        sigma, mu = 0.5, 0.0
        g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
        # Convert to torch tensor with channel dimension
        img = torch.from_numpy(g.astype(np.float32)).float().unsqueeze(0)
        return img

def display_results(original_img, transformed_imgs, applied_transforms_list, num_rows=3):
    """Display the original and transformed images with transform information."""
    num_examples = len(transformed_imgs)
    num_cols = 3  # Original + 2 transformed per row
    
    # Calculate center slice index
    center_slice = original_img.shape[-1] // 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    
    # Helper function to get center slice
    def get_center_slice(img):
        return img[0, :, :, center_slice].cpu().numpy()
    
    # Plot original in first column of each row
    for row in range(num_rows):
        ax = axes[row, 0] if num_rows > 1 else axes[0]
        ax.imshow(get_center_slice(original_img), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')
    
    # Plot transformed images
    for i, (img, applied) in enumerate(zip(transformed_imgs, applied_transforms_list)):
        row = i // 2
        col = (i % 2) + 1
        
        if row < num_rows:
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.imshow(get_center_slice(img), cmap='gray')
            
            # Create title with applied transforms
            applied_names = [name for name, was_applied in applied.items() if was_applied]
            title = f"Example {i+1}\n"
            # Add applied transforms (up to 3, then "...")
            if applied_names:
                title += ", ".join(applied_names[:3])
                if len(applied_names) > 3:
                    title += f", +{len(applied_names)-3} more"
            else:
                title += "No transforms applied"
                
            ax.set_title(title, fontsize=9)
            ax.axis('off')
    
    plt.tight_layout()
    output_dir = Path("domain_rand_test_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(output_dir / f"domain_rand_test_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.show()

def log_applied_transforms(iteration, applied_transforms):
    """Log which transforms were applied in each iteration."""
    applied_names = [name for name, was_applied in applied_transforms.items() if was_applied]
    logger.info(f"Iteration {iteration}: Applied {len(applied_names)}/{len(applied_transforms)} transforms")
    
    if applied_names:
        logger.info(f"  Applied: {', '.join(applied_names)}")
    else:
        logger.info("  No transforms were applied")

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise

def test_domain_randomization(config_path, num_iterations=10, image_path=None):
    """Test domain randomization by applying it multiple times to the same image."""
    logger.info(f"Starting domain randomization test using config from {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    domain_rand_config = config.get("domain_randomization", {})
    
    # Check if domain randomization is enabled
    if not domain_rand_config.get("use_domain_randomization", False):
        logger.warning("Domain randomization is disabled in the config file. Enabling it for testing.")
        domain_rand_config["use_domain_randomization"] = True
    
    # Device setup (use config device if specified, otherwise auto-detect)
    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Load sample image
    if image_path:
        # Use the specified image path
        logger.info(f"Using specified image: {image_path}")
        img = load_sample_image(image_path)
    else:
        # Try to find an existing NIfTI file, or use synthetic data if not found
        sample_paths = [
            "data/sample_brain.nii.gz",  # Check for sample data in data directory
            "brain_age_pred/tests/data/sample_brain.nii.gz",  # Check tests directory
        ]
        
        img = None
        for path in sample_paths:
            if os.path.exists(path):
                logger.info(f"Loading sample image from {path}")
                img = load_sample_image(path)
                break
        
        if img is None:
            logger.info("No sample image found, creating synthetic data")
            img = load_sample_image(None)  # This will create synthetic data
    
    # Move image to device
    img = img.to(device)
    original_img = img.clone()
    
    # Extract parameters from config
    image_key = domain_rand_config.get("image_key", "image")
    
    # Create the tracking domain randomizer
    randomizer = TrackingDomainRandomizer(
        device=device,
        **domain_rand_config  # Pass all domain randomization config parameters
    )
    
    logger.info("Domain randomizer initialized with these probabilities:")
    for name, prob in randomizer.prob.items():
        logger.info(f"  {name}: {prob:.2f}")
    
    # Apply domain randomization multiple times
    transformed_imgs = []
    applied_transforms_list = []
    
    # Create a sample like what would be used in real training
    # This simulates the structure from dataset.py's __getitem__
    sample = {
        image_key: original_img,
        "age": torch.tensor(55.0, dtype=torch.float32),  # Example age
        "weight": torch.tensor(1.0, dtype=torch.float32)  # Example sample weight
    }
    
    for i in range(num_iterations):
        logger.info(f"Applying domain randomization iteration {i+1}/{num_iterations}")
        
        # Clone the sample to avoid modifying the original
        current_sample = {
            image_key: sample[image_key].clone(),
            "age": sample["age"].clone(),
            "weight": sample["weight"].clone()
        }
        
        # Apply randomization
        start_time = time.time()
        result = randomizer(current_sample)
        end_time = time.time()
        
        # Get the transformed image
        transformed_img = result[image_key]
        transformed_imgs.append(transformed_img.clone())
        
        # Get which transforms were applied
        applied_transforms = randomizer.get_applied_transforms()
        applied_transforms_list.append(applied_transforms)
        
        # Log results
        log_applied_transforms(i+1, applied_transforms)
        logger.info(f"  Processing time: {end_time - start_time:.3f} seconds")
    
    # Display the results
    display_results(original_img, transformed_imgs, applied_transforms_list)
    
    # Create summary statistics
    logger.info("\nSummary Statistics:")
    transform_counts = {name: 0 for name in randomizer.prob.keys()}
    
    for applied in applied_transforms_list:
        for name, was_applied in applied.items():
            if was_applied:
                transform_counts[name] = transform_counts.get(name, 0) + 1
    
    for name, count in transform_counts.items():
        frequency = count / num_iterations
        logger.info(f"  {name}: applied {count}/{num_iterations} times ({frequency:.1%})")
    
    logger.info("Domain randomization test completed")

if __name__ == "__main__":
    config_path = "C:/Projects/thesis_project/brain_age_pred/configs/sfcn/sfcn_dom_rand_tuning.yaml"
    image_path = "C:/Projects/thesis_project/Data/brain_age_preprocessed/CamCAN/sub-CC110045_T1w.nii.gz"
    test_domain_randomization(config_path, num_iterations=10, image_path=image_path)