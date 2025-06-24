"""
Test file for the BABrainGenerator that generates multiple synthetic brain examples.
This test demonstrates the brain generator's capability to create domain-randomized
synthetic brain volumes from segmentation maps.
"""

import os
import sys
import unittest
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
import tempfile
import shutil
import matplotlib.pyplot as plt

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.brain_gen.brain_generator import BABrainGenerator, DEFAULT_GENERATION_LABELS, DEFAULT_N_NEUTRAL_LABELS


class TestBrainGenerator(unittest.TestCase):
    """Test class for BABrainGenerator functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.project_root = Path(__file__).parent.parent
        cls.data_dir = cls.project_root / "data"
        
        # Available segmentation paths
        cls.seg_paths = {
            "seg_T1": cls.data_dir / "templates" / "seg_T1.nii.gz",
            "seg_T2": cls.data_dir / "templates" / "seg_T2.nii.gz", 
            "seg_18_40": cls.data_dir / "templates" / "seg_18_40.nii.gz",
            "seg_40_60": cls.data_dir / "templates" / "seg_40_60.nii.gz",
            "seg_60_85": cls.data_dir / "templates" / "seg_60_85.nii.gz",
            "seg_FLAIR": cls.data_dir / "templates" / "seg_FLAIR.nii.gz",
        }
        
        # Default SynthSeg prior parameters for intensity generation
        # These are example values - adjust based on your data
        cls.prior_means = np.array([
            [25, 100, 75, 0, 125, 100, 125, 100, 125, 100, 125, 100, 125, 100],  # mean locations
            [5, 15, 10, 1, 15, 10, 15, 10, 15, 10, 15, 10, 15, 10]               # mean scales
        ])
        
        cls.prior_stds = np.array([
            [5, 10, 8, 1, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8],    # std locations
            [2, 5, 3, 0.5, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3]        # std scales
        ])
        
        # Augmentation probabilities
        cls.aug_prob = {
            "flip": 0.5,
            "affine": 0.8,
            "contrast": 0.7,
            "gamma": 0.7,
            "scale_int": 0.6,
            "shift_int": 0.6,
            "hist_shift": 0.5,
            "noise": 0.6,
            "rician": 0.3,
            "gibbs": 0.2,
            "blur": 0.4,
            "bias": 0.6,
            "resolution": 0.8,
        }
        
        # Create temporary output directory for test results
        cls.temp_dir = tempfile.mkdtemp(prefix="brain_gen_test_")
        print(f"Test outputs will be saved to: {cls.temp_dir}")
    
    @classmethod 
    def tearDownClass(cls):
        """Clean up test fixtures after running tests"""
        # Clean up temporary directory
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def create_brain_generator(self, **kwargs):
        """Create a BABrainGenerator with default parameters"""
        default_params = {
            "prior_means": self.prior_means,
            "prior_stds": self.prior_stds,
            "distribution": "uniform",
            "prob": self.aug_prob,
            "rotation_range": 15.0,
            "scaling_range": 0.15,
            "shearing_bounds": 0.1,
            "translation_bounds": 5.0,
            "contrast_range": (0.8, 1.2),
            "log_gamma_std": 0.1,
            "shift_offset": 0.1,
            "hist_control_points": 5,
            "noise_mean": 0.0,
            "noise_std": 0.05,
            "rician_std": 0.03,
            "gibbs_alpha": (0.5, 1.0),
            "blur_sigma": (0.5, 1.5),
            "bias_field_rng": (0.0, 0.3),
            "min_res": 1.0,
            "max_res_iso": 4.0,
            "max_res_aniso": 8.0,
            "atlas_res": 1.0,
            "generation_labels": DEFAULT_GENERATION_LABELS,
            "n_neutral_labels": DEFAULT_N_NEUTRAL_LABELS,
            "use_hemisphere_aware_flip": True,
            "use_dynamic_resolution": True,
            "use_intensity_clip_normalize": True,
            "intensity_clip_value": 300.0,
            "intensity_gamma_std": 0.5,
        }
        default_params.update(kwargs)
        return BABrainGenerator(**default_params)
    
    def load_segmentation(self, seg_path):
        """Load a segmentation file and prepare it for the generator"""
        if not seg_path.exists():
            self.skipTest(f"Segmentation file not found: {seg_path}")
        
        print(f"Loading segmentation: {seg_path}")
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata().astype(np.int32)
        
        # Convert to tensor and add batch/channel dimensions
        seg_tensor = torch.from_numpy(seg_data).unsqueeze(0)  # Add channel dimension
        
        print(f"Segmentation shape: {seg_tensor.shape}")
        print(f"Unique labels: {torch.unique(seg_tensor).numpy()}")
        
        return seg_tensor, seg_img.affine, seg_img.header
    
    def save_generated_image(self, image_tensor, affine, header, output_path):
        """Save generated image to NIfTI file"""
        # Convert tensor to numpy and remove channel dimension
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.squeeze().cpu().numpy()
        else:
            image_np = np.squeeze(image_tensor)
        
        # Create NIfTI image and save
        nii_img = nib.Nifti1Image(image_np, affine=affine, header=header)
        nib.save(nii_img, output_path)
        print(f"Saved generated image: {output_path}")
    
    def test_single_channel_generation(self):
        """Test single-channel brain generation from different segmentations"""
        print("\n=== Testing Single Channel Generation ===")
        
        generator = self.create_brain_generator(n_channels=1)
        
        for seg_name, seg_path in list(self.seg_paths.items())[:3]:  # Test first 3
            with self.subTest(segmentation=seg_name):
                print(f"\nTesting with {seg_name}")
                
                # Load segmentation
                seg_tensor, affine, header = self.load_segmentation(seg_path)
                
                # Prepare sample
                sample = {"image": seg_tensor}
                
                # Generate synthetic brain
                result = generator(sample)
                
                # Verify output
                self.assertIn("image", result)
                generated_image = result["image"]
                self.assertIsInstance(generated_image, torch.Tensor)
                self.assertEqual(generated_image.shape[1:], seg_tensor.shape[1:])  # Same spatial dims
                
                # Save result
                output_path = Path(self.temp_dir) / f"generated_{seg_name}_single.nii.gz"
                self.save_generated_image(generated_image, affine, header, output_path)
    
    def test_multi_channel_generation(self):
        """Test multi-channel brain generation"""
        print("\n=== Testing Multi-Channel Generation ===")
        
        n_channels = 3
        generator = self.create_brain_generator(
            n_channels=n_channels,
            use_specific_stats_for_channel=False
        )
        
        # Use T1 segmentation for multi-channel test
        seg_path = self.seg_paths["seg_T1"]
        seg_tensor, affine, header = self.load_segmentation(seg_path)
        
        sample = {"image": seg_tensor}
        result = generator(sample)
        
        # Verify multi-channel output
        generated_image = result["image"]
        self.assertEqual(generated_image.shape[0], n_channels)
        
        # Save each channel separately
        for ch in range(n_channels):
            channel_image = generated_image[ch:ch+1]  # Keep channel dimension
            output_path = Path(self.temp_dir) / f"generated_T1_channel_{ch}.nii.gz"
            self.save_generated_image(channel_image, affine, header, output_path)
    
    def test_different_augmentation_settings(self):
        """Test generation with different augmentation settings"""
        print("\n=== Testing Different Augmentation Settings ===")
        
        # Test with minimal augmentations
        minimal_prob = {key: 0.1 for key in self.aug_prob.keys()}
        generator_minimal = self.create_brain_generator(prob=minimal_prob)
        
        # Test with heavy augmentations  
        heavy_prob = {key: 0.9 for key in self.aug_prob.keys()}
        generator_heavy = self.create_brain_generator(prob=heavy_prob)
        
        seg_path = self.seg_paths["seg_18_40"]
        seg_tensor, affine, header = self.load_segmentation(seg_path)
        sample = {"image": seg_tensor}
        
        # Generate with minimal augmentations
        result_minimal = generator_minimal(sample.copy())
        output_path = Path(self.temp_dir) / "generated_minimal_aug.nii.gz"
        self.save_generated_image(result_minimal["image"], affine, header, output_path)
        
        # Generate with heavy augmentations
        result_heavy = generator_heavy(sample.copy())
        output_path = Path(self.temp_dir) / "generated_heavy_aug.nii.gz"
        self.save_generated_image(result_heavy["image"], affine, header, output_path)
        
        # Verify different outputs
        self.assertFalse(torch.allclose(result_minimal["image"], result_heavy["image"]))
    
    def test_age_specific_segmentations(self):
        """Test generation using age-specific segmentations"""
        print("\n=== Testing Age-Specific Segmentations ===")
        
        age_groups = ["seg_18_40", "seg_40_60", "seg_60_85"]
        generator = self.create_brain_generator()
        
        for age_group in age_groups:
            with self.subTest(age_group=age_group):
                print(f"\nTesting {age_group}")
                
                seg_path = self.seg_paths[age_group]
                seg_tensor, affine, header = self.load_segmentation(seg_path)
                
                sample = {"image": seg_tensor}
                result = generator(sample)
                
                # Save result
                output_path = Path(self.temp_dir) / f"generated_{age_group}.nii.gz"
                self.save_generated_image(result["image"], affine, header, output_path)
    
    def test_multiple_examples_same_segmentation(self):
        """Test generating multiple different examples from the same segmentation"""
        print("\n=== Testing Multiple Examples from Same Segmentation ===")
        
        generator = self.create_brain_generator()
        seg_path = self.seg_paths["seg_T1"]
        seg_tensor, affine, header = self.load_segmentation(seg_path)
        
        n_examples = 5
        generated_images = []
        
        for i in range(n_examples):
            sample = {"image": seg_tensor}
            result = generator(sample)
            generated_images.append(result["image"])
            
            # Save each example
            output_path = Path(self.temp_dir) / f"generated_T1_example_{i+1}.nii.gz"
            self.save_generated_image(result["image"], affine, header, output_path)
        
        # Verify all examples are different (stochastic generation)
        for i in range(n_examples):
            for j in range(i+1, n_examples):
                self.assertFalse(
                    torch.allclose(generated_images[i], generated_images[j], rtol=1e-3),
                    f"Examples {i+1} and {j+1} are too similar"
                )
    
    def test_gradient_computation(self):
        """Test brain generation with gradient computation enabled"""
        print("\n=== Testing Gradient Computation ===")
        
        generator = self.create_brain_generator(return_gradients=True)
        seg_path = self.seg_paths["seg_T1"]
        seg_tensor, affine, header = self.load_segmentation(seg_path)
        
        sample = {"image": seg_tensor}
        result = generator(sample)
        
        # Check if gradients were computed (this depends on the ImageGradientsD implementation)
        self.assertIn("image", result)
        print(f"Result keys: {result.keys()}")
        
        # Save result
        output_path = Path(self.temp_dir) / "generated_with_gradients.nii.gz"
        self.save_generated_image(result["image"], affine, header, output_path)
    
    def test_custom_output_shape(self):
        """Test generation with custom output shape (cropping)"""
        print("\n=== Testing Custom Output Shape ===")
        
        output_shape = (128, 128, 128)  # Smaller than typical brain volumes
        generator = self.create_brain_generator(
            output_shape=output_shape,
            use_random_cropping=True
        )
        
        seg_path = self.seg_paths["seg_T1"]
        seg_tensor, affine, header = self.load_segmentation(seg_path)
        
        sample = {"image": seg_tensor}
        result = generator(sample)
        
        # Verify output shape matches requested shape
        generated_image = result["image"]
        self.assertEqual(generated_image.shape[1:], output_shape)
        
        # Save result
        output_path = Path(self.temp_dir) / "generated_custom_shape.nii.gz"
        self.save_generated_image(generated_image, affine, header, output_path)


def run_generation_examples():
    """Standalone function to run brain generation examples"""
    print("Running Brain Generation Examples...")
    
    # Run the test suite
    unittest.main(argv=[''], exit=False, verbosity=2)


def create_brain_generator():
    """Create a BABrainGenerator with default parameters"""
    
    # Get the actual labels from the segmentation first
    data_dir = project_root / "data"
    seg_path = data_dir / "templates" / "seg_T1.nii.gz"
    
    if seg_path.exists():
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata().astype(np.int32)
        unique_labels = np.unique(seg_data)
        max_label = int(np.max(unique_labels))
        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        print(f"Maximum label value: {max_label}")
    else:
        # Fallback to default
        unique_labels = DEFAULT_GENERATION_LABELS
        max_label = int(np.max(unique_labels))
    
    # Create prior parameters that can handle the maximum label value
    # We need arrays of size (max_label + 1) to handle indexing by label value
    array_size = max_label + 1
    
    # Initialize with default values
    means = np.zeros(array_size)
    stds = np.ones(array_size)
    
    # Define realistic intensity values for common brain structures
    # These are based on typical T1-weighted MRI intensities
    intensity_map = {
        0: 0,    # background
        2: 80,   # left cerebral white matter
        3: 60,   # left cerebral cortex
        4: 30,   # left lateral ventricle
        5: 50,   # left inf lat vent
        7: 50,   # left cerebellar white matter
        8: 60,   # left cerebellar cortex
        10: 50,  # left thalamus proper
        11: 50,  # left caudate
        12: 50,  # left putamen
        13: 50,  # left pallidum
        14: 30,  # 3rd ventricle
        15: 30,  # 4th ventricle
        16: 70,  # brain stem
        17: 50,  # left hippocampus
        18: 50,  # left amygdala
        24: 30,  # CSF
        26: 50,  # left accumbens area
        28: 50,  # left ventral DC
        41: 80,  # right cerebral white matter
        42: 60,  # right cerebral cortex
        43: 30,  # right lateral ventricle
        44: 50,  # right inf lat vent
        46: 50,  # right cerebellar white matter
        47: 60,  # right cerebellar cortex
        49: 50,  # right thalamus proper
        50: 50,  # right caudate
        51: 50,  # right putamen
        52: 50,  # right pallidum
        53: 50,  # right hippocampus
        54: 50,  # right amygdala
        58: 50,  # right accumbens area
        60: 50,  # right ventral DC
    }
    
    std_map = {
        0: 1,    # background
        2: 12,   # white matter
        3: 10,   # cortex
        4: 5,    # ventricles
        5: 8,    # inf lat vent
        7: 12,   # cerebellar white matter
        8: 10,   # cerebellar cortex
        10: 8,   # thalamus
        11: 8,   # caudate
        12: 8,   # putamen
        13: 8,   # pallidum
        14: 5,   # 3rd ventricle
        15: 5,   # 4th ventricle
        16: 10,  # brain stem
        17: 8,   # hippocampus
        18: 8,   # amygdala
        24: 5,   # CSF
        26: 8,   # accumbens
        28: 8,   # ventral DC
        41: 12,  # right white matter
        42: 10,  # right cortex
        43: 5,   # right ventricle
        44: 8,   # right inf lat vent
        46: 12,  # right cerebellar white matter
        47: 10,  # right cerebellar cortex
        49: 8,   # right thalamus
        50: 8,   # right caudate
        51: 8,   # right putamen
        52: 8,   # right pallidum
        53: 8,   # right hippocampus
        54: 8,   # right amygdala
        58: 8,   # right accumbens
        60: 8,   # right ventral DC
    }
    
    # Fill in the arrays
    for label in unique_labels:
        label = int(label)
        means[label] = intensity_map.get(label, 50)  # default to 50 if not found
        stds[label] = std_map.get(label, 8)          # default to 8 if not found
    
    # Prior parameters for intensity generation (2 rows: [means, scales])
    prior_means = np.array([
        means,                    # mean locations
        stds / 10                 # mean scales (smaller values)
    ])
    
    prior_stds = np.array([
        stds / 2,                 # std locations
        stds / 20                 # std scales
    ])
    
    print(f"Prior means shape: {prior_means.shape}")
    print(f"Prior stds shape: {prior_stds.shape}")
    print(f"Array size to handle max label {max_label}: {array_size}")
    
    # Augmentation probabilities
    aug_prob = {
        "flip": 0.5,
        "affine": 0.8,
        "contrast": 0.7,
        "gamma": 0.7,
        "scale_int": 0.6,
        "shift_int": 0.6,
        "hist_shift": 0.5,
        "noise": 0.6,
        "rician": 0.3,
        "gibbs": 0.2,
        "blur": 0.4,
        "bias": 0.6,
        "resolution": 0.8,
    }
    
    generator = BABrainGenerator(
        prior_means=prior_means,
        prior_stds=prior_stds,
        distribution="uniform",
        prob=aug_prob,
        rotation_range=15.0,
        scaling_range=0.15,
        shearing_bounds=0.1,
        translation_bounds=5.0,
        contrast_range=(0.8, 1.2),
        log_gamma_std=0.1,
        shift_offset=0.1,
        hist_control_points=5,
        noise_mean=0.0,
        noise_std=0.05,
        rician_std=0.03,
        gibbs_alpha=(0.5, 1.0),
        blur_sigma=(0.5, 1.5),
        bias_field_rng=(0.0, 0.3),
        min_res=1.0,
        max_res_iso=4.0,
        max_res_aniso=8.0,
        atlas_res=1.0,
        generation_labels=unique_labels,
        n_neutral_labels=min(7, len(unique_labels)),
        use_hemisphere_aware_flip=True,
        use_dynamic_resolution=True,
        use_intensity_clip_normalize=True,
        intensity_clip_value=300.0,
        intensity_gamma_std=0.5,
    )
    
    return generator


def load_segmentation(seg_path):
    """Load a segmentation file and prepare it for the generator"""
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
    
    print(f"Loading segmentation: {seg_path}")
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata().astype(np.int32)
    
    # Convert to tensor and add channel dimension
    seg_tensor = torch.from_numpy(seg_data).unsqueeze(0)  # Add channel dimension
    
    print(f"Segmentation shape: {seg_tensor.shape}")
    print(f"Unique labels: {torch.unique(seg_tensor).numpy()}")
    
    return seg_tensor, seg_img.affine, seg_img.header


def generate_brain_examples():
    """Generate 10 different brain images from the same segmentation and display them"""
    
    # Set up paths
    data_dir = project_root / "data"
    seg_path = data_dir / "templates" / "seg_T1.nii.gz"  # Use T1 segmentation
    
    # Check if segmentation exists
    if not seg_path.exists():
        print(f"Segmentation file not found: {seg_path}")
        print("Available template files:")
        templates_dir = data_dir / "templates"
        if templates_dir.exists():
            for file in templates_dir.glob("seg_*.nii.gz"):
                print(f"  - {file.name}")
        return
    
    # Create generator
    print("Creating brain generator...")
    generator = create_brain_generator()
    
    # Load segmentation
    seg_tensor, affine, header = load_segmentation(seg_path)
    
    # Generate 10 different examples
    print("\nGenerating 10 different brain examples...")
    generated_images = []
    
    for i in range(10):
        print(f"Generating example {i+1}/10...")
        sample = {"image": seg_tensor}
        result = generator(sample)
        
        # Convert to numpy for visualization
        generated_image = result["image"].squeeze().cpu().numpy()
        generated_images.append(generated_image)
    
    # Display the results
    print("\nDisplaying results...")
    plot_brain_examples(generated_images, seg_tensor.squeeze().numpy())


def plot_brain_examples(generated_images, original_seg):
    """Plot the generated brain examples in a grid"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('10 Different Synthetic Brain Images from Same Segmentation', fontsize=16)
    
    # Plot original segmentation in first subplot
    axes[0, 0].imshow(original_seg[:, :, original_seg.shape[2]//2], cmap='tab20', aspect='equal')
    axes[0, 0].set_title('Original Segmentation')
    axes[0, 0].axis('off')
    
    # Plot generated images
    for i, img in enumerate(generated_images):
        row = (i + 1) // 4
        col = (i + 1) % 4
        
        # Show middle axial slice
        slice_idx = img.shape[2] // 2
        axes[row, col].imshow(img[:, :, slice_idx], cmap='gray', aspect='equal')
        axes[row, col].set_title(f'Generated #{i+1}')
        axes[row, col].axis('off')
    
    # Hide the last subplot (we have 11 total: 1 original + 10 generated)
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Also create a comparison showing different slices of one example
    fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
    fig2.suptitle('Different Slices of Generated Brain Example #1', fontsize=16)
    
    example_img = generated_images[0]
    
    # Show axial slices
    for i in range(5):
        slice_idx = int((i + 1) * example_img.shape[2] / 6)  # Different slices
        axes2[0, i].imshow(example_img[:, :, slice_idx], cmap='gray', aspect='equal')
        axes2[0, i].set_title(f'Axial Slice {slice_idx}')
        axes2[0, i].axis('off')
    
    # Show sagittal slices
    for i in range(5):
        slice_idx = int((i + 1) * example_img.shape[0] / 6)  # Different slices
        axes2[1, i].imshow(example_img[slice_idx, :, :], cmap='gray', aspect='equal')
        axes2[1, i].set_title(f'Sagittal Slice {slice_idx}')
        axes2[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Generated image shape: {generated_images[0].shape}")
    print(f"Intensity range: [{np.min(generated_images[0]):.2f}, {np.max(generated_images[0]):.2f}]")
    print(f"Mean intensity: {np.mean(generated_images[0]):.2f}")
    print(f"Standard deviation: {np.std(generated_images[0]):.2f}")
    
    # Check if images are different
    print("\nVerifying stochastic generation:")
    for i in range(1, len(generated_images)):
        similarity = np.corrcoef(generated_images[0].flatten(), generated_images[i].flatten())[0, 1]
        print(f"Correlation between example 1 and {i+1}: {similarity:.3f}")


if __name__ == "__main__":
    try:
        generate_brain_examples()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()