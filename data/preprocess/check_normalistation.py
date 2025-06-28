#!/usr/bin/env python3
"""
check_mri_npy.py
Quick QA script for 3-D brain MRI volumes saved as .npy files.

Checks performed on each file:
1. Shape                  → exactly (182, 218, 182)       (Axial × Coronal × Sagittal)
2. Dtype                  → float32 or float64
3. Finite values          → no NaN / ±inf
4. Intensity distribution →
   • mean in [-1, 1]
   • std  in [0.1, 2.5]
   • min  ≥ -5 and max ≤ 5
   (Assumes z-score normalisation; edit limits to match your pipeline)
5. Non-empty voxels       → at least 1 % of voxels are non-zero
6. Optional: symmetry check (L/R sums within 5 %)

A per-file PASS / FAIL report is printed and a summary tallied at the end.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

SHAPE_EXPECTED = (182, 218, 182)             # EDIT if you use a different template
DTYPE_ALLOWED  = {np.float32, np.float64}

# Intensity limits – adapt to match your normalisation strategy
MEAN_RANGE = (-1.0, 1.0)
STD_RANGE  = (0.1, 2.5)
MIN_LIM    = -5.0
MAX_LIM    =  5.0

console = Console()


def check_volume(arr: np.ndarray, fname: str) -> list[str]:
    """Return a list of error messages; empty list means PASS."""
    errs = []

    # 1. shape
    if arr.shape != SHAPE_EXPECTED:
        errs.append(f"shape {arr.shape} (expected {SHAPE_EXPECTED})")

    # 2. dtype
    if arr.dtype not in DTYPE_ALLOWED:
        errs.append(f"dtype {arr.dtype} (expected float32/64)")

    # 3. finite
    if not np.isfinite(arr).all():
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        errs.append(f"non-finite values (nan={n_nan}, inf={n_inf})")

    # Stop here if non-finite; everything else relies on finite values
    if errs:
        return errs

    # 4. intensity stats
    mu  = float(arr.mean())
    std = float(arr.std())
    mn, mx = float(arr.min()), float(arr.max())

    if not MEAN_RANGE[0] <= mu <= MEAN_RANGE[1]:
        errs.append(f"mean {mu:.3f} outside {MEAN_RANGE}")
    if not STD_RANGE[0] <= std <= STD_RANGE[1]:
        errs.append(f"std {std:.3f} outside {STD_RANGE}")
    if mn < MIN_LIM or mx > MAX_LIM:
        errs.append(f"min/max [{mn:.3f}, {mx:.3f}] outside [{MIN_LIM}, {MAX_LIM}]")

    # 5. sparsity
    nonzero_ratio = np.count_nonzero(arr) / arr.size
    if nonzero_ratio < 0.01:
        errs.append(f"only {100*nonzero_ratio:.2f}% voxels non-zero (<1%)")

    # 6. (optional) left-right symmetry crude heuristic
    # Split on sagittal mid-plane (axis=2)
    if arr.shape == SHAPE_EXPECTED:
        mid = arr.shape[2] // 2
        left_sum  = arr[:, :, :mid].sum()
        right_sum = arr[:, :, mid:].sum()
        if abs(left_sum - right_sum) / (abs(left_sum) + 1e-8) > 0.05:
            errs.append("L/R integral differs by >5% (possible mis-registration)")

    return errs


def main():
    parser = argparse.ArgumentParser(description="QA check for .npy MR volumes")
    parser.add_argument("folder", type=Path, help="directory containing *.npy files")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="scan sub-directories as well")
    args = parser.parse_args()

    if not args.folder.exists():
        console.print(f"[bold red]Folder {args.folder} does not exist[/]")
        sys.exit(1)

    pattern = "**/*.npy" if args.recursive else "*.npy"
    files = sorted(args.folder.glob(pattern))
    if not files:
        console.print(f"[bold red]No .npy files found in {args.folder}[/]")
        sys.exit(1)

    table = Table(show_lines=False)
    table.add_column("File", style="cyan", overflow="fold")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    n_pass = n_fail = 0
    for f in files:
        try:
            vol = np.load(f)
        except Exception as e:
            n_fail += 1
            table.add_row(str(f), "[red]FAIL[/]", f"could not load – {e}")
            continue

        errors = check_volume(vol, f.name)
        if errors:
            table.add_row(str(f), "[red]FAIL[/]", "; ".join(errors))
            n_fail += 1
        else:
            table.add_row(str(f), "[green]PASS[/]", "✓")
            n_pass += 1

    console.print(table)
    console.print(f"[bold green]PASS[/]: {n_pass}    [bold red]FAIL[/]: {n_fail}    "
                  f"Total: {len(files)}")


if __name__ == "__main__":
    main()