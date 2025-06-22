#!/usr/bin/env python
"""
Pre-process Nifti volumes (denoise → N4 → intensity scale) and save as NumPy arrays.

Example
-------
python nifti_preprocess_to_numpy.py \
    --input  /path/to/nifti  \
    --output /path/to/numpy  \
    --no_denoise            # <-- optional switches
    --no_n4
    --scale_min 0   --scale_max 1
    --num_workers 8 --verbose

Dependencies
------------
pip install antspyx monai nibabel tqdm
"""

import argparse
import logging
import multiprocessing
import time
from functools import partial
from pathlib import Path

import ants
import nibabel as nib   # used only to catch exotic file types; can be removed
import numpy as np
from monai.transforms import ScaleIntensity
from tqdm import tqdm


# -----------------------------------------------------------------------------#
# Utility functions
# -----------------------------------------------------------------------------#
def setup_logger(verbose: bool = False) -> logging.Logger:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("nifti_preprocessor")


def run_denoise(img: ants.core.ants_image.ANTsImage) -> ants.core.ants_image.ANTsImage:
    """Rician denoising (ANTs)."""
    return ants.denoise_image(img, noise_model="Rician")


def run_n4(img: ants.core.ants_image.ANTsImage) -> ants.core.ants_image.ANTsImage:
    """N4 bias–field correction (ANTs)."""
    return ants.n4_bias_field_correction(img)


# -----------------------------------------------------------------------------#
# Core worker
# -----------------------------------------------------------------------------#
def process_file(
    nifti_path: Path,
    rel_root: Path,
    out_root: Path,
    *,
    apply_denoise: bool,
    apply_n4: bool,
    scale_min: float,
    scale_max: float,
    verbose: bool = False,
) -> bool:
    """
    1. Load Nifti with ANTs
    2. Denoise  (optional)
    3. N4       (optional)
    4. Intensity scaling to [scale_min, scale_max]
    5. Save .npy mirroring directory structure
    """
    log = logging.getLogger("nifti_preprocessor")

    try:
        # ------------------- load ------------------------------------------------
        img_ants = ants.image_read(str(nifti_path))
        if img_ants is None:
            raise RuntimeError("ANTs failed to read image")

        # ------------------- preprocessing --------------------------------------
        if apply_denoise:
            log.debug(f"Denoising {nifti_path.name}")
            img_ants = run_denoise(img_ants)

        if apply_n4:
            log.debug(f"N4 bias-field correction {nifti_path.name}")
            img_ants = run_n4(img_ants)

        # to numpy
        img_np = img_ants.numpy().astype(np.float32)

        # ------------------- intensity scaling ----------------------------------
        scaler = ScaleIntensity(minv=scale_min, maxv=scale_max)
        img_np = scaler(img_np)

        # ------------------- write ----------------------------------------------
        rel_path = nifti_path.relative_to(rel_root).with_suffix(".npy")
        out_path = out_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, img_np)

        return True

    except Exception as e:
        if verbose:
            log.exception(f"Failed on {nifti_path}: {e}")
        return False


# -----------------------------------------------------------------------------#
# File discovery
# -----------------------------------------------------------------------------#
def find_nifti_files(root: Path, extensions) -> list[Path]:
    files = []
    for ext in extensions:
        files.extend(root.glob(f"**/*{ext}"))
    return files


# -----------------------------------------------------------------------------#
# Driver
# -----------------------------------------------------------------------------#
def convert_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    extensions,
    num_workers: int,
    apply_denoise: bool,
    apply_n4: bool,
    scale_min: float,
    scale_max: float,
    verbose: bool,
):
    log = setup_logger(verbose)

    nifti_files = find_nifti_files(input_dir, extensions)
    if not nifti_files:
        log.warning("No Nifti files found. Exiting.")
        return

    log.info(f"Found {len(nifti_files)} files. Starting conversion …")
    t0 = time.time()

    worker_fn = partial(
        process_file,
        rel_root=input_dir,
        out_root=output_dir,
        apply_denoise=apply_denoise,
        apply_n4=apply_n4,
        scale_min=scale_min,
        scale_max=scale_max,
        verbose=verbose,
    )

    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_fn, nifti_files),
                    total=len(nifti_files),
                    desc="Processing",
                )
            )
    else:
        results = [
            worker_fn(f) for f in tqdm(nifti_files, desc="Processing", total=len(nifti_files))
        ]

    ok = sum(results)
    log.info(f"Finished in {time.time()-t0:.1f}s  ({ok}/{len(results)} succeeded)")


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser(description="Nifti → NumPy with denoise / N4 / scaling")
    parser.add_argument("--input", required=True, help="Input directory with Nifti data")
    parser.add_argument("--output", required=True, help="Output directory for .npy files")
    parser.add_argument(
        "--extensions",
        default=".nii,.nii.gz",
        help="Comma-separated list of file extensions",
    )

    # Processing switches
    parser.add_argument("--no_denoise", action="store_true", help="Skip denoising step")
    parser.add_argument("--no_n4", action="store_true", help="Skip N4 bias-field correction")
    parser.add_argument(
        "--scale_min",
        type=float,
        default=0.0,
        help="Minimum value after intensity scaling (default 0)",
    )
    parser.add_argument(
        "--scale_max",
        type=float,
        default=1.0,
        help="Maximum value after intensity scaling (default 1)",
    )

    # Runtime
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    convert_batch(
        Path(args.input).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        extensions=[e.strip() for e in args.extensions.split(",")],
        num_workers=args.num_workers,
        apply_denoise=not args.no_denoise,
        apply_n4=not args.no_n4,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()