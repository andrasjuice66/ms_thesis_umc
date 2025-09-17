#!/usr/bin/env python
"""
Brain Generator Spectating Script
Saves synthetic images generated from the brain generator for inspection.
"""
import os, sys, time, json, random
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch
import nibabel as nib

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from brain_age_pred.configs.config import Config
from brain_age_pred.utils.logger import setup_logger
from brain_age_pred.utils.utils import set_seed, read_csv
from brain_age_pred.brain_gen.brain_generator import BABrainGenerator
from brain_age_pred.brain_gen.labels import GENERATION_LABELS, GENERATION_CLASSES, N_NEUTRAL_LABELS


def save_image_as_nifti(image_tensor, save_path, affine=None):
    """Save a torch tensor as NIfTI file using nibabel."""
    # Convert tensor to numpy and remove channel dimension if present
    if len(image_tensor.shape) == 4:  # (C, D, H, W)
        image_np = image_tensor.squeeze(0).cpu().numpy()
    else:  # (D, H, W)
        image_np = image_tensor.cpu().numpy()
    
    # Use identity matrix if no affine provided
    if affine is None:
        affine = np.eye(4)
    
    # Create NIfTI image using nibabel and save
    nifti_img = nib.Nifti1Image(image_np, affine)
    nib.save(nifti_img, str(save_path))


def save_image_as_numpy(image_tensor, save_path):
    """Save a torch tensor as numpy array (.npy file)."""
    # Convert tensor to numpy and remove channel dimension if present
    if len(image_tensor.shape) == 4:  # (C, D, H, W)
        image_np = image_tensor.squeeze(0).cpu().numpy()
    else:  # (D, H, W)
        image_np = image_tensor.cpu().numpy()
    
    # Save as numpy array
    np.save(str(save_path), image_np)


def generate_and_save_images(brain_generator, segmentation_paths, output_dir, prefix, num_images=10, save_format="nifti", batch_size=1, random_ages=False):
    """Generate synthetic images and save them for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Generating and saving {num_images} {prefix} images to {output_dir} (format: {save_format}) ===")
    
    # Limit to requested number of images
    segmentation_paths = segmentation_paths[:num_images]
    
    metadata = []
    
    for i, seg_path in enumerate(segmentation_paths):
        try:
            # Load segmentation image
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata()
            
            # Convert to torch tensor
            seg_tensor = torch.from_numpy(seg_data).long()
            
            # Generate random age if requested, otherwise use middle age (50)
            if random_ages:
                age = random.uniform(20, 80)
            else:
                age = 50.0
            
            # Prepare input for brain generator
            sample = {
                "image": seg_tensor,
                "age": torch.tensor([age]),
                "__image_path__": str(seg_path)
            }
            
            # Apply brain generator transform
            result = brain_generator(sample)
            
            # Get the generated image
            generated_image = result["image"]
            
            # Create filename with metadata
            original_name = Path(seg_path).stem
            if save_format.lower() == "nifti":
                filename = f"{prefix}_{i:03d}_{original_name}_age{age:.1f}.nii.gz"
                save_path = output_dir / filename
                save_image_as_nifti(generated_image, save_path, affine=seg_nii.affine)
            elif save_format.lower() == "numpy":
                filename = f"{prefix}_{i:03d}_{original_name}_age{age:.1f}.npy"
                save_path = output_dir / filename
                save_image_as_numpy(generated_image, save_path)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
            
            # Save segmentation too if it's in the result
            if "seg_gt" in result:
                seg_filename = f"{prefix}_{i:03d}_{original_name}_seg_gt.nii.gz"
                seg_save_path = output_dir / seg_filename
                save_image_as_nifti(result["seg_gt"], seg_save_path, affine=seg_nii.affine)
            
            # Store metadata
            metadata.append({
                "index": i,
                "filename": filename,
                "age": float(age),
                "original_path": str(seg_path),
                "image_shape": list(generated_image.shape),
                "image_min": float(generated_image.min()),
                "image_max": float(generated_image.max()),
                "image_mean": float(generated_image.mean()),
                "image_std": float(generated_image.std()),
                "save_format": save_format,
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{len(segmentation_paths)} images...")
                
        except Exception as e:
            print(f"  Error generating image from {seg_path}: {e}")
            continue
    
    # Save metadata as JSON
    metadata_path = output_dir / f"{prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Completed! Generated {len(metadata)} images and saved metadata to {output_dir}")
    return metadata


def main():
    # 1. ─── configuration & reproducibility ─────────────────── #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic/brainagenext_synthetic.yaml"
    cfg = Config(cfg_file)
    
    # 2. ─── experiment naming / I/O ─────────────────────────── #
    experiment_name = cfg.get("output.experiment_name")
    if not experiment_name:
        experiment_name = f'brain_gen_debug_{timestamp}'
    
    out_root = Path("debug_output")
    spectate_dir = out_root / experiment_name
    spectate_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("brain-gen-debug", log_file=spectate_dir / "debug.log")
    
    logger.info("Initializing brain generator debug...")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Experiment: {experiment_name}\nConfig: {cfg_file}")
    logger.info(f"Output directory: {spectate_dir}")

    # 3. ─── Set up device ─────────────────────────────────────── #
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 4. ─── Initialize Brain Generator ────────────────────────── #
    logger.info("Initializing Brain Generator...")
    bg_cfg = cfg.get("brain_generator", {})
    
    # Augmentation probabilities from config
    prob = bg_cfg.get("prob", {})
    
    # Prior distribution parameters
    mean_loc = bg_cfg.get("mean_loc", 125.0)
    mean_scale = bg_cfg.get("mean_scale", 100.0)
    std_loc = bg_cfg.get("std_loc", 15.0)
    std_scale = bg_cfg.get("std_scale", 10.0)
    
    n_classes = len(GENERATION_LABELS)

    # "loc" = mid-point, "scale" = half-range
    prior_means = np.vstack([
        np.full(n_classes, mean_loc, dtype=float),
        np.full(n_classes, mean_scale, dtype=float),
    ])

    prior_stds = np.vstack([
        np.full(n_classes, std_loc, dtype=float),
        np.full(n_classes, std_scale, dtype=float),
    ])

    # Set background class (label 0) to zero
    prior_means[:, 0] = 0.0    
    prior_stds[:, 0] = 0.0

    # Initialize brain generator
    brain_generator = BABrainGenerator(
        # Required parameters
        prior_means=prior_means,
        prior_stds=prior_stds,
        distribution=bg_cfg.get("distribution", "normal"),
        prob=prob,

        # Spatial augmentation parameters
        rotation_range=bg_cfg.get("rotation_range", 10),
        scaling_range=bg_cfg.get("scaling_range", 0.1),
        shear_bounds=bg_cfg.get("shear_bounds", 0.005),
        translation_bounds=bg_cfg.get("translation_bounds", False),

        # Intensity augmentation parameters
        contrast_range=tuple(bg_cfg.get("contrast_range", [0.8, 1.2])),
        log_gamma_std=bg_cfg.get("log_gamma_std", 0.1),
        shift_offset=bg_cfg.get("shift_offset", 0.1),
        hist_control_points=bg_cfg.get("hist_control_points", 5),

        # Artefacts parameters
        noise_mean=bg_cfg.get("noise_mean", 0.02),
        noise_std=bg_cfg.get("noise_std", 0.015),
        rician_std=bg_cfg.get("rician_std", 0.01),
        gibbs_alpha=bg_cfg.get("gibbs_alpha", 0.4),
        blur_sigma=bg_cfg.get("blur_sigma", 0.25),
        bias_field_rng=tuple(bg_cfg.get("bias_field_rng", [0.0, 0.5])),
        
        # Motion artifacts
        motion_degrees=bg_cfg.get("motion_degrees", 3),
        motion_translation=bg_cfg.get("motion_translation", 5),
        motion_num_transforms=bg_cfg.get("motion_num_transforms", 4),
        ghost_num=tuple(bg_cfg.get("ghost_num", [1, 4])),
        ghost_intensity=tuple(bg_cfg.get("ghost_intensity", [0.1, 0.6])),
        torchio_noise_std=bg_cfg.get("torchio_noise_std", [0, 0.5]),

        # Resolution parameters
        min_res=bg_cfg.get("min_res", 0.8),
        max_res_iso=bg_cfg.get("max_res_iso", 2.0),
        max_res_aniso=bg_cfg.get("max_res_aniso", 2.0),
        atlas_res=bg_cfg.get("atlas_res", 1.0),
        thickness=bg_cfg.get("thickness", None),

        # Label config parameters
        generation_labels=GENERATION_LABELS,
        n_neutral_labels=N_NEUTRAL_LABELS,
        output_labels=None,

        # Toggle parameters
        use_sample=bg_cfg.get("use_sample", True),
        use_hemisphere_aware_flip=bg_cfg.get("use_hemisphere_aware_flip", True),
        use_dynamic_resolution=bg_cfg.get("use_dynamic_resolution", True),
        use_intensity_clip_normalize=bg_cfg.get("use_intensity_clip_normalize", True),
        n_channels=bg_cfg.get("n_channels", 1),
        use_specific_stats_for_channel=bg_cfg.get("use_specific_stats_for_channel", False),
        output_shape=tuple(bg_cfg.get("output_shape", [160, 192, 160])),
        use_random_cropping=bg_cfg.get("use_random_cropping", True),
        return_gradients=bg_cfg.get("return_gradients", False),
        return_segmentation=True,  # Always return segmentation for visualization
        device=device,
    )

    # 5. ─── Prepare Input Segmentations ─────────────────────── #
    logger.info("Reading CSV files to get segmentation paths...")
    segmented_data_dir = Path(cfg.get("data.segmented_data_dir"))
    train_csv = Path(cfg.get("data.train_csv"))
    
    logger.info(f"Reading train CSV from {train_csv}")
    train_p, train_a, train_w, train_s, train_m = read_csv(
        train_csv,
        segmented_data_dir,
    )

    # 6. ─── Generate and Save Images ───────────────────────── #
    logger.info("Generating and saving synthetic images...")
    
    # Save config for reference
    cfg.save_config(spectate_dir / "config.yaml")
    
    # Determine save format
    save_format = cfg.get("spectate.save_format", "nifti")  # "nifti" or "numpy"
    
    # Generate multiple sets with different parameters
    # Standard generations
    standard_metadata = generate_and_save_images(
        brain_generator=brain_generator,
        segmentation_paths=train_p,
        output_dir=spectate_dir / "standard",
        prefix="standard",
        num_images=10,
        save_format=save_format,
        random_ages=True
    )
    
    # With tumor (if supported)
    if "tumor" in prob:
        # Temporarily increase tumor probability to 1.0
        old_tumor_prob = prob.get("tumor", 0.0)
        prob["tumor"] = 1.0
        
        tumor_metadata = generate_and_save_images(
            brain_generator=brain_generator,
            segmentation_paths=train_p,
            output_dir=spectate_dir / "with_tumor",
            prefix="tumor",
            num_images=10,
            save_format=save_format,
            random_ages=True
        )
        
        # Restore original probability
        prob["tumor"] = old_tumor_prob
    
    # With motion artifacts (if supported)
    if "motion" in prob:
        # Temporarily increase motion probability to 1.0
        old_motion_prob = prob.get("motion", 0.0)
        prob["motion"] = 1.0
        
        motion_metadata = generate_and_save_images(
            brain_generator=brain_generator,
            segmentation_paths=train_p,
            output_dir=spectate_dir / "with_motion",
            prefix="motion",
            num_images=10,
            save_format=save_format,
            random_ages=True
        )
        
        # Restore original probability
        prob["motion"] = old_motion_prob
    
    # With ghosting artifacts (if supported)
    if "ghost" in prob:
        # Temporarily increase ghost probability to 1.0
        old_ghost_prob = prob.get("ghost", 0.0)
        prob["ghost"] = 1.0
        
        ghost_metadata = generate_and_save_images(
            brain_generator=brain_generator,
            segmentation_paths=train_p,
            output_dir=spectate_dir / "with_ghost",
            prefix="ghost",
            num_images=10,
            save_format=save_format,
            random_ages=True
        )
        
        # Restore original probability
        prob["ghost"] = old_ghost_prob
    
    # With noise artifacts (if supported)
    if "torchio_noise" in prob:
        # Temporarily increase noise probability to 1.0
        old_noise_prob = prob.get("torchio_noise", 0.0)
        prob["torchio_noise"] = 1.0
        
        noise_metadata = generate_and_save_images(
            brain_generator=brain_generator,
            segmentation_paths=train_p,
            output_dir=spectate_dir / "with_noise",
            prefix="noise",
            num_images=10,
            save_format=save_format,
            random_ages=True
        )
        
        # Restore original probability
        prob["torchio_noise"] = old_noise_prob
    
    # 7. ─── summary ───────────────────────────────────────────── #
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config_file": cfg_file,
        "save_format": save_format,
        "brain_generator_config": bg_cfg,
        "output_directory": str(spectate_dir),
    }
    
    summary_path = spectate_dir / "debug_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=== BRAIN GENERATOR DEBUGGING COMPLETE ===")
    logger.info(f"Images saved to: {spectate_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    
    print(f"\n🎉 Brain generator debug complete!")
    print(f"📁 Check your images in: {spectate_dir}")
    print(f"📊 Summary: {summary_path}")
    print(f"💾 Format: {save_format}")


if __name__ == "__main__":
    sys.exit(main())