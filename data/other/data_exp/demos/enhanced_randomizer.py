"""
Domain randomization for brain MRI images using TorchIO and MONAI.
Specifically designed for creating augmented datasets for brain age prediction models.
"""

import os
import torch
import numpy as np
import torchio as tio
import monai
from monai.transforms import (
    Compose, 
    RandAffined,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    ToTensord
)
from monai.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path


class BrainAgeRandomizer:
    """
    Domain randomization for brain MRI images to create diverse datasets for age prediction.
    Separates different types of transformations for clarity and modularity.
    """
    
    def __init__(
        self,
        data_dir,
        image_key="image",
        age_key="age",
        
        # Output parameters
        batchsize=4,
        n_channels=1,
        target_res=None,
        output_shape=None,
        
        # Intensity parameters
        intensity_bounds=(0, 100),
        contrast_bounds=(0.75, 1.25),
        prior_distributions='uniform',
        
        # Spatial deformation parameters
        flipping=True,
        scaling_bounds=0.2,
        rotation_bounds=15,
        shearing_bounds=0.012,
        translation_bounds=False,
        nonlin_std=3.0,
        
        # Blurring/resampling parameters
        randomise_res=True,
        max_res_iso=3.0,
        max_res_aniso=5.0,
        
        # Bias field parameters
        bias_field_std=0.5,
        
        # Other parameters
        num_samples=1,
        output_dir="randomized_samples",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the brain age randomizer with configurable parameters.
        """
        # Basic parameters
        self.data_dir = data_dir
        self.image_key = image_key
        self.age_key = age_key
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.device = device
        
        # Output parameters
        self.batchsize = batchsize
        self.n_channels = n_channels
        self.target_res = target_res
        self.output_shape = output_shape
        
        # Intensity parameters
        self.intensity_bounds = intensity_bounds
        self.contrast_bounds = contrast_bounds
        self.prior_distributions = prior_distributions
        
        # Spatial deformation parameters
        self.flipping = flipping
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.nonlin_std = nonlin_std
        
        # Blurring/resampling parameters
        self.randomise_res = randomise_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        
        # Bias field parameters
        self.bias_field_std = bias_field_std
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data paths
        self.data_paths = self._get_data_paths()
        
        # Initialize transforms
        self._initialize_transforms()
        
    def _get_data_paths(self):
        """Get paths to all MRI images and their corresponding age labels."""
        data_paths = []
        image_dir = os.path.join(self.data_dir, "images")
        
        print(f"Looking for images in: {image_dir}")
        if not os.path.exists(image_dir):
            print(f"ERROR: Image directory does not exist: {image_dir}")
            return data_paths
        
        # Assuming age labels are stored in a CSV file
        age_file = os.path.join(self.data_dir, "age_labels.csv")
        print(f"Looking for age labels in: {age_file}")
        
        age_dict = {}
        
        # Load age data if file exists
        if os.path.exists(age_file):
            import csv
            with open(age_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    subject_id, age = row
                    age_dict[subject_id] = float(age)
            print(f"Found {len(age_dict)} subjects with age labels")
        else:
            print(f"ERROR: Age labels file not found: {age_file}")
        
        image_count = 0
        matched_count = 0
        
        for img_file in os.listdir(image_dir):
            if img_file.endswith((".nii.gz", ".nii")):
                image_count += 1
                img_path = os.path.join(image_dir, img_file)
                
                # Extract subject ID correctly from filename
                # For files like "sub01.nii.gz", we want "sub01"
                subject_id = img_file
                if subject_id.endswith('.nii.gz'):
                    subject_id = subject_id[:-7]  # Remove .nii.gz
                elif subject_id.endswith('.nii'):
                    subject_id = subject_id[:-4]  # Remove .nii
                    
                print(f"Image file: {img_file}, extracted subject ID: {subject_id}")
                    
                if subject_id in age_dict:
                    matched_count += 1
                    data_paths.append({
                        self.image_key: img_path,
                        self.age_key: age_dict[subject_id]
                    })
                else:
                    print(f"WARNING: No age found for subject {subject_id}")
        
        print(f"Found {image_count} image files, matched {matched_count} with age labels")
        print(f"Final dataset size: {len(data_paths)}")
        
        return data_paths
    
    def _initialize_transforms(self):
        """Initialize all transformation components separately for clarity."""
        # Initialize different types of transforms
        self.spatial_transforms = self._get_spatial_transforms()
        self.intensity_transforms = self._get_intensity_transforms()
        self.resolution_transforms = self._get_resolution_transforms()
        self.artifact_transforms = self._get_artifact_transforms()
        
        # Combine all transforms
        self.transform = self._combine_transforms()
    
    def _get_spatial_transforms(self):
        """Create spatial transformations (affine, elastic deformations)."""
        # Process parameters to match expected format
        if isinstance(self.scaling_bounds, (int, float)):
            scale_range = (1 - self.scaling_bounds, 1 + self.scaling_bounds)
        elif self.scaling_bounds is False:
            scale_range = (1.0, 1.0)
        else:
            scale_range = self.scaling_bounds
            
        if isinstance(self.rotation_bounds, (int, float)):
            rot_range = (np.pi * self.rotation_bounds / 180.0,) * 3
        else:
            rot_range = np.array(self.rotation_bounds) * np.pi / 180.0
            
        if isinstance(self.shearing_bounds, (int, float)):
            shear_range = self.shearing_bounds
        elif self.shearing_bounds is False:
            shear_range = 0.0
        else:
            shear_range = self.shearing_bounds
            
        if isinstance(self.translation_bounds, (int, float)):
            translate_range = (self.translation_bounds,) * 3
        elif self.translation_bounds is False:
            translate_range = (0.0,) * 3
        else:
            translate_range = self.translation_bounds
        
        # TorchIO spatial transforms
        torchio_spatial = []
        
        # Add elastic deformation if nonlin_std > 0
        if self.nonlin_std > 0:
            torchio_spatial.append(
                tio.RandomElasticDeformation(
                    max_displacement=self.nonlin_std,
                    p=0.7
                )
            )
        
        # MONAI spatial transforms
        monai_spatial = [
            RandAffined(
                keys=[self.image_key],
                prob=0.8,
                rotate_range=rot_range,
                scale_range=scale_range,
                shear_range=shear_range,
                translate_range=translate_range,
                mode="bilinear",
                padding_mode="zeros"
            )
        ]
        
        return {
            "torchio": tio.Compose(torchio_spatial),
            "monai": Compose(monai_spatial)
        }
    
    def _get_intensity_transforms(self):
        """Create intensity transformations (contrast, brightness)."""
        # Intensity transforms based on distribution type
        if self.prior_distributions == 'uniform':
            intensity_transform = RandGaussianNoised(
                keys=[self.image_key],
                prob=0.7,
                mean=0.0,
                std=self.intensity_bounds[1]/10  # Use a single value
            )
            contrast_transform = RandAdjustContrastd(
                keys=[self.image_key],
                prob=0.7,
                gamma=self.contrast_bounds[0]  # Use a single value
            )
        else:  # 'normal'
            intensity_transform = RandGaussianNoised(
                keys=[self.image_key],
                prob=0.7,
                mean=0.0,
                std=self.intensity_bounds[1]/20
            )
            contrast_transform = RandAdjustContrastd(
                keys=[self.image_key],
                prob=0.7,
                gamma=(self.contrast_bounds[0] + self.contrast_bounds[1])/2
            )
        
        monai_intensity = [
            intensity_transform,
            contrast_transform
        ]
        
        return {
            "monai": Compose(monai_intensity)
        }
    
    def _get_resolution_transforms(self):
        """Create resolution transformations (blur, downsampling)."""
        torchio_resolution = []
        
        # Add resolution randomization if enabled
        if self.randomise_res:
            # For isotropic resolution randomization
            if self.max_res_iso is not None:
                torchio_resolution.append(
                    tio.RandomBlur(
                        std=(0.1, self.max_res_iso / 2),
                        p=0.7
                    )
                )
            
            # For anisotropic resolution randomization
            if self.max_res_aniso is not None:
                torchio_resolution.append(
                    tio.RandomAnisotropy(
                        axes=(0, 1, 2),
                        downsampling=(1, self.max_res_aniso / 2),
                        p=0.4
                    )
                )
        
        # MONAI resolution transforms
        monai_resolution = [
            RandGaussianSmoothd(
                keys=[self.image_key],
                prob=0.5,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5)
            )
        ]
        
        return {
            "torchio": tio.Compose(torchio_resolution),
            "monai": Compose(monai_resolution)
        }
    
    def _get_artifact_transforms(self):
        """Create MRI-specific artifact transformations (bias field, ghosting)."""
        torchio_artifacts = []
        
        # Add bias field if bias_field_std > 0
        if self.bias_field_std > 0:
            torchio_artifacts.append(
                tio.RandomBiasField(
                    coefficients=self.bias_field_std,
                    order=3,
                    p=0.7
                )
            )
            
        # Add other MRI-specific artifacts
        torchio_artifacts.extend([
            tio.RandomGhosting(
                num_ghosts=(2, 7),
                axes=(0, 1, 2),
                intensity=(0.2, 0.8),
                p=0.3
            ),
            tio.RandomSpike(
                num_spikes=(1, 3),
                intensity=(0.1, 0.3),
                p=0.2
            )
        ])
        
        # MONAI artifact transforms
        monai_artifacts = [
            RandBiasFieldd(
                keys=[self.image_key],
                prob=0.7,
                coeff_range=(0.1, self.bias_field_std)
            )
        ]
        
        return {
            "torchio": tio.Compose(torchio_artifacts),
            "monai": Compose(monai_artifacts)
        }
    
    def _combine_transforms(self):
        """Combine all transforms into a single pipeline."""
        # Final conversion to tensor
        self.final_transform = Compose([
            ToTensord(keys=[self.image_key])
        ])
        
        # Return the combined transform method
        return self.combined_transform_method

    def combined_transform_method(self, data):
        """Apply all transforms in sequence."""
        # Create a copy of the data to avoid modifying the original
        data_dict = dict(data)
        
        # Apply TorchIO transforms to the image
        subject = tio.Subject({
            self.image_key: tio.ScalarImage(data_dict[self.image_key])
        })
        
        # Apply spatial transforms (TorchIO)
        subject = self.spatial_transforms["torchio"](subject)
        
        # Apply resolution transforms (TorchIO)
        subject = self.resolution_transforms["torchio"](subject)
        
        # Apply artifact transforms (TorchIO)
        subject = self.artifact_transforms["torchio"](subject)
        
        # Convert back to dictionary for MONAI transforms
        data_dict[self.image_key] = subject[self.image_key].data.numpy()
        
        # Apply spatial transforms (MONAI)
        data_dict = self.spatial_transforms["monai"](data_dict)
        
        # Apply intensity transforms (MONAI)
        data_dict = self.intensity_transforms["monai"](data_dict)
        
        # Apply resolution transforms (MONAI)
        data_dict = self.resolution_transforms["monai"](data_dict)
        
        # Apply artifact transforms (MONAI)
        data_dict = self.artifact_transforms["monai"](data_dict)
        
        # Apply final transforms
        data_dict = self.final_transform(data_dict)
        
        return data_dict
    
    def create_dataset(self):
        """Create a MONAI dataset with the domain randomization transforms."""
        return Dataset(
            data=self.data_paths,
            transform=self.transform
        )
    
    def create_dataloader(self, shuffle=True, num_workers=0):
        """Create a DataLoader from the randomized dataset."""
        dataset = self.create_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batchsize,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def generate_samples(self, save_visualizations=True):
        """Generate randomized samples from the dataset for visualization."""
        dataset = self.create_dataset()
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0  # Use 0 workers to avoid pickling issues
        )
        
        for i, batch in enumerate(dataloader):
            if i >= self.num_samples:
                break
                
            # Get the randomized image and age
            image = batch[self.image_key].squeeze().cpu().numpy()
            age = batch[self.age_key].item()
            
            # Save the randomized samples
            sample_dir = os.path.join(self.output_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save as numpy array (simple and reliable)
            np.save(os.path.join(sample_dir, "image.npy"), image)
            
            # Save age information
            with open(os.path.join(sample_dir, "age.txt"), "w") as f:
                f.write(f"Age: {age:.1f} years")
            
            # Visualize if requested
            if save_visualizations:
                self._visualize_sample(image, age, sample_dir)
                
            print(f"Generated sample {i+1}/{self.num_samples}")
    
    def _visualize_sample(self, image, age, sample_dir):
        """Visualize a slice of the randomized image with age information."""
        # Choose a middle slice for visualization
        if len(image.shape) == 3:
            # Create a multi-slice view (axial, sagittal, coronal)
            slice_ax = image[:, :, image.shape[2]//2]
            slice_sag = image[:, image.shape[1]//2, :]
            slice_cor = image[image.shape[0]//2, :, :]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot axial slice
            axes[0].imshow(slice_ax, cmap='gray')
            axes[0].set_title('Axial')
            axes[0].axis('off')
            
            # Plot sagittal slice
            axes[1].imshow(slice_sag, cmap='gray')
            axes[1].set_title('Sagittal')
            axes[1].axis('off')
            
            # Plot coronal slice
            axes[2].imshow(slice_cor, cmap='gray')
            axes[2].set_title('Coronal')
            axes[2].axis('off')
            
            plt.suptitle(f"Brain Age: {age:.1f} years", fontsize=14)
        else:
            # Handle 2D case
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Brain Age: {age:.1f} years")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'visualization.png'))
        plt.close()


def main():
    """Main function to demonstrate the brain age randomizer."""
    # Example usage
    data_dir = "/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/DataExp/data"  # Replace with your data directory
    
    # Initialize the brain age randomizer
    randomizer = BrainAgeRandomizer(
        data_dir=data_dir,
        batchsize=4,
        n_channels=1,
        
        # Intensity parameters
        intensity_bounds=(0, 80),
        contrast_bounds=(0.8, 1.2),
        
        # Spatial deformation parameters
        scaling_bounds=0.15,
        rotation_bounds=10,
        nonlin_std=3.0,
        
        # Blurring/resampling parameters
        randomise_res=True,
        max_res_iso=3.0,
        
        # Bias field parameters
        bias_field_std=0.4,
        
        # Other parameters
        num_samples=1,
        output_dir="brain_age_samples"
    )
    
    # Generate randomized samples for visualization
    randomizer.generate_samples(save_visualizations=True)
    
    # Create a dataloader for training a model (not doing the training here)
    dataloader = randomizer.create_dataloader(shuffle=True)
    print(f"Created dataloader with {len(dataloader)} batches")
    
    print("Brain age dataset preparation complete!")


if __name__ == "__main__":
    main() 