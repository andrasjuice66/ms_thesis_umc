#!/usr/bin/env python3
"""
batch_segmentation.py
-----------------------------------------------
Run SynthSeg on all NIfTI volumes in a directory structure
and recreate the same directory structure with segmentation outputs.
"""

import argparse
import subprocess
import time
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import wandb
from tqdm import tqdm


def run_synthseg(in_img: Path, out_dir: Path, threads: int = 4, use_gpu: bool = False) -> Path:
    """
    Launch SynthSeg through Docker.
    The function returns the path to the generated segmentation.
    """
    # Keep the same filename for the segmentation output
    seg_path = out_dir / in_img.name
    # CSV file with same name but .csv extension
    vol_path = out_dir / f"{in_img.stem}.csv"

    # Convert paths to Docker-compatible format
    in_dir = in_img.parent
    out_dir_abs = out_dir.absolute()
    in_dir_abs = in_dir.absolute()

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["docker", "run", "--rm"]
    
    # Add GPU support if requested
    if use_gpu:
        cmd.extend(["--gpus", "all"])
    
    cmd.extend([
        "-v", f"{in_dir_abs}:/input",
        "-v", f"{out_dir_abs}:/output",
        "cookpa/synthseg:conda-0.1",  # You may need a GPU-enabled version
        "--i", f"/input/{in_img.name}",
        "--o", f"/output/{in_img.name}",
        "--vol", f"/output/{in_img.stem}.csv",
        "--robust",         # robust intensity normalisation
        "--threads", str(threads)
    ])
    
    print(f"[SynthSeg] Processing {in_img} {'(GPU)' if use_gpu else '(CPU)'}")
    print("[SynthSeg] " + " ".join(cmd))
    
    try:
        subprocess.check_call(cmd)
        print(f"[SynthSeg] Successfully processed {in_img}")
        return seg_path
    except subprocess.CalledProcessError as e:
        print(f"[SynthSeg] Error processing {in_img}: {e}")
        raise


def find_nifti_files(input_dir: Path) -> List[Path]:
    """
    Recursively find all .nii.gz files in the input directory.
    """
    nifti_files = []
    for file_path in input_dir.rglob("*.nii.gz"):
        if file_path.is_file():
            nifti_files.append(file_path)
    
    print(f"Found {len(nifti_files)} .nii.gz files")
    return nifti_files


def get_relative_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    """
    Get the corresponding output path maintaining directory structure.
    """
    # Get relative path from input root
    relative_path = input_file.relative_to(input_root)
    
    # Create corresponding path in output directory
    output_path = output_root / relative_path.parent
    
    return output_path


def setup_wandb(project_name: str = "synthseg-batch-processing") -> None:
    """
    Setup wandb with API key login and initialize run.
    """
    print("Setting up wandb logging...")
    
    # Login to wandb using API key
    wandb.login(key = "2abdb867a9244072f2237704a3cacc77fa548dd8")
    
    # Initialize wandb run
    wandb.init(
        project=project_name,
        config={
            "tool": "SynthSeg",
            "processing_type": "batch_segmentation"
        }
    )
    
    print("✓ wandb initialized successfully")


def main(input_dir, output_dir) -> None:
    # Hardcoded configuration values
    threads = 6
    use_gpu = False
    wandb_project = "thesis_preprocess"
    use_wandb = True
    
    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup wandb if enabled
    if use_wandb:
        try:
            setup_wandb(wandb_project)
        except Exception as e:
            print(f"Warning: Failed to setup wandb: {e}")
            print("Continuing without wandb logging...")
            use_wandb = False
    
    # Find all NIfTI files
    print("Scanning for .nii.gz files...")
    nifti_files = find_nifti_files(input_dir)
    
    if not nifti_files:
        print("No .nii.gz files found in the input directory")
        if use_wandb:
            wandb.finish()
        return
    
    print(f"Found {len(nifti_files)} .nii.gz files")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using {threads} threads")
    print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Log initial info to wandb
    if use_wandb:
        wandb.config.update({
            "total_files": len(nifti_files),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "threads": threads,
            "gpu_acceleration": use_gpu
        })
    
    print("-" * 50)
    
    # Process each file with progress bar
    successful_files = 0
    failed_files = 0
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(
        nifti_files, 
        desc="Processing files", 
        unit="file",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for i, nifti_file in enumerate(pbar, 1):
        # Update progress bar description
        pbar.set_description(f"Processing {nifti_file.name}")
        
        try:
            # Get corresponding output directory
            file_output_dir = get_relative_output_path(nifti_file, input_dir, output_dir)
            
            # Run SynthSeg with GPU option
            seg_path = run_synthseg(nifti_file, file_output_dir, threads=threads, use_gpu=use_gpu)
            
            successful_files += 1
            
            # Log progress to wandb
            if use_wandb:
                wandb.log({
                    "files_processed": i,
                    "files_successful": successful_files,
                    "files_failed": failed_files,
                    "progress_percent": (i / len(nifti_files)) * 100,
                    "current_file": nifti_file.name
                })
            
        except Exception as e:
            failed_files += 1
            pbar.write(f"[Error] Failed to process {nifti_file}: {e}")
            
            # Log error to wandb
            if use_wandb:
                wandb.log({
                    "files_processed": i,
                    "files_successful": successful_files,
                    "files_failed": failed_files,
                    "progress_percent": (i / len(nifti_files)) * 100,
                    "error": str(e),
                    "failed_file": nifti_file.name
                })
            
            continue
    
    # Close progress bar
    pbar.close()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    success_rate = (successful_files / len(nifti_files)) * 100 if nifti_files else 0
    
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files: {len(nifti_files)}")
    print(f"Successful: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per file: {total_time/len(nifti_files):.1f} seconds")
    print(f"Output directory: {output_dir}")
    
    # Log final summary to wandb
    if use_wandb:
        wandb.log({
            "final_total_files": len(nifti_files),
            "final_successful": successful_files,
            "final_failed": failed_files,
            "final_success_rate": success_rate,
            "total_processing_time": total_time,
            "avg_time_per_file": total_time/len(nifti_files) if nifti_files else 0
        })
        
        # Create a summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Total Files", len(nifti_files)],
                ["Successful", successful_files],
                ["Failed", failed_files],
                ["Success Rate (%)", f"{success_rate:.1f}"],
                ["Total Time (s)", f"{total_time:.1f}"],
                ["Avg Time per File (s)", f"{total_time/len(nifti_files):.1f}" if nifti_files else "0"]
            ]
        )
        wandb.log({"processing_summary": summary_table})
        
        print("✓ Results logged to wandb")
        wandb.finish()


if __name__ == "__main__":
    # input_dir = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/CoRR")
    # output_dir = Path("/mnt/c/Projects/thesis_project/Data/brain_age_segmented/CoRR")
    # main(input_dir, output_dir)
    # input_dir = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/PanGen")
    # output_dir = Path("/mnt/c/Projects/thesis_project/Data/brain_age_segmented/PanGen")
    # main(input_dir, output_dir)
    # input_dir = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/SALD")
    # output_dir = Path("/mnt/c/Projects/thesis_project/Data/brain_age_segmented/SALD")
    # main(input_dir, output_dir)

    input_dir = Path("/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/AOMIC_ID1000")
    output_dir = Path("/mnt/c/Projects/thesis_project/Data/brain_age_segmented/AOMIC_ID1000")
    main(input_dir, output_dir)


"""
python3 segmentation_dir.py "/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed/" "/mnt/c/Projects/thesis_project/Data/brain_age_pred/brain_age_segmented/" --threads 6 --wandb-project "thesis_preprocess"
"""