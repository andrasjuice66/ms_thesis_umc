#!/usr/bin/env python
"""
Convert Nifti image files to normalized NumPy arrays while preserving directory structure.

Usage:
    python nifti_to_numpy.py --input /path/to/nifti/data --output /path/to/numpy/output [options]

Options:
    --input INPUT       Input directory containing Nifti files (recursive search)
    --output OUTPUT     Output directory for NumPy files (will create if doesn't exist)
    --normalize         Apply percentile-based normalization (default: True)
    --percentile_low    Lower percentile for normalization (default: 1)
    --percentile_high   Upper percentile for normalization (default: 99)
    --extensions        Comma-separated list of Nifti extensions to process (default: .nii,.nii.gz)
    --num_workers       Number of parallel workers (default: 4)
    --verbose           Print detailed information during processing
"""

import os
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from functools import partial
import logging
import time

def setup_logger(verbose=False):
    """Set up logging."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("nifti_converter")

def process_file(file_path, output_path, normalize=True, perc_low=1, perc_high=99, verbose=False):
    """Process a single Nifti file and save as normalized NumPy array."""
    try:
        # Load the Nifti file
        img = nib.load(str(file_path)).get_fdata(dtype=np.float32)
        
        # Apply normalization if requested
        if normalize:
            vmin, vmax = np.percentile(img, (perc_low, perc_high))
            img = np.clip(img, vmin, vmax)
            img = (img - vmin) / (vmax - vmin + 1e-6)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as NumPy array
        np.save(output_path, img)
        return True
    except Exception as e:
        if verbose:
            print(f"Error processing {file_path}: {str(e)}")
        return False

def worker_task(args):
    """Worker function for multiprocessing."""
    nifti_file, output_dir, input_dir, normalize, perc_low, perc_high, verbose = args
    
    # Construct output path with same relative structure
    rel_path = nifti_file.relative_to(input_dir)
    output_file = output_dir / rel_path.with_suffix('.npy')
    
    return process_file(nifti_file, output_file, normalize, perc_low, perc_high, verbose)

def find_nifti_files(input_dir, extensions):
    """Find all Nifti files with the specified extensions."""
    input_path = Path(input_dir)
    all_files = []
    
    for ext in extensions:
        all_files.extend(list(input_path.glob(f'**/*{ext}')))
    
    return all_files

def convert_nifti_to_numpy(input_dir, output_dir, normalize=True, 
                          perc_low=1, perc_high=99, extensions=None,
                          num_workers=4, verbose=False):
    """
    Convert all Nifti files in input_dir to normalized NumPy arrays in output_dir.
    Preserves the directory structure from input_dir.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing Nifti files (will be searched recursively)
    output_dir : str or Path
        Directory where NumPy files will be saved
    normalize : bool, default=True
        Whether to apply percentile-based normalization
    perc_low : int, default=1
        Lower percentile for normalization
    perc_high : int, default=99
        Upper percentile for normalization
    extensions : list, default=['.nii', '.nii.gz']
        List of Nifti file extensions to search for
    num_workers : int, default=4
        Number of parallel workers for processing
    verbose : bool, default=False
        Whether to print detailed information
    """
    logger = setup_logger(verbose)
    
    # Set up paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if extensions is None:
        extensions = ['.nii', '.nii.gz']
    
    # Find all Nifti files
    logger.info(f"Searching for Nifti files in {input_path}")
    nifti_files = find_nifti_files(input_path, extensions)
    
    if not nifti_files:
        logger.warning(f"No Nifti files found in {input_path} with extensions {extensions}")
        return
    
    logger.info(f"Found {len(nifti_files)} Nifti files to process")
    
    # Prepare arguments for parallel processing
    worker_args = [
        (nifti_file, output_path, input_path, normalize, perc_low, perc_high, verbose)
        for nifti_file in nifti_files
    ]
    
    # Process files
    start_time = time.time()
    
    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_task, worker_args),
                total=len(worker_args),
                desc="Converting files"
            ))
    else:
        results = []
        for args in tqdm(worker_args, desc="Converting files"):
            results.append(worker_task(args))
    
    # Calculate statistics
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    elapsed_time = time.time() - start_time
    
    logger.info(f"Conversion completed in {elapsed_time:.2f} seconds")
    logger.info(f"Successfully converted: {successful} files")
    if failed > 0:
        logger.warning(f"Failed to convert: {failed} files")

def main():
    parser = argparse.ArgumentParser(description='Convert Nifti files to normalized NumPy arrays')
    parser.add_argument('--input', required=True, help='Input directory containing Nifti files')
    parser.add_argument('--output', required=True, help='Output directory for NumPy files')
    parser.add_argument('--normalize', action='store_true', default=True, 
                        help='Apply percentile-based normalization')
    parser.add_argument('--percentile_low', type=int, default=1,
                        help='Lower percentile for normalization')
    parser.add_argument('--percentile_high', type=int, default=99,
                        help='Upper percentile for normalization')
    parser.add_argument('--extensions', type=str, default='.nii,.nii.gz',
                        help='Comma-separated list of Nifti extensions to process')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during processing')
    
    args = parser.parse_args()
    
    # Parse comma-separated extensions
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    
    convert_nifti_to_numpy(
        args.input,
        args.output,
        normalize=args.normalize,
        perc_low=args.percentile_low,
        perc_high=args.percentile_high,
        extensions=extensions,
        num_workers=args.num_workers,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
