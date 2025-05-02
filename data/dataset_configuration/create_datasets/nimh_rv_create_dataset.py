#!/usr/bin/env python3
"""
Build a manifest that links every pre-processed image to the
participant's sex and age.

Output columns
---------------
subject_id , image_path , sex , age , modality , dataset , eligibility (if available)

Usage
-----
python create_image_manifest.py \
    --participants participants.tsv \
    --input-root  /mnt/data/NIMH_RV \
    --output      image_manifest.tsv \
    --dataset     nimh_rv

If you already have a *text* file that simply lists the full paths
(one per line) you can pass it with `--paths-list` instead of
`--input-root`.
"""
import argparse
import pathlib
import re
import sys
import os
import shutil

import pandas as pd


# ---------------------------------------------------------------------
def guess_modality(path: str) -> str:
    """
    Return 't1', 't2' or 'flair' based on the filename.
    """
    fname = pathlib.Path(path).name.lower()
    if "_t1w" in fname:
        return "t1"
    if "_t2w" in fname:
        return "t2"
    if "_flair" in fname:
        return "flair"
    return "unknown"


SUB_RE = re.compile(r"(sub-[a-zA-Z0-9]+)")


def extract_subject(path: str) -> str:
    """
    Grab the first `sub-XXXXX` token from the path.
    """
    m = SUB_RE.search(path)
    if not m:
        raise ValueError(f"No subject id found in: {path}")
    return m.group(1)


# ---------------------------------------------------------------------
def collect_paths(root: pathlib.Path) -> list[str]:
    """
    Recursively gather all .nii or .nii.gz files under *root*.
    """
    return [
        str(p)
        for p in root.rglob("*.nii*")
    ]


# ---------------------------------------------------------------------
def build_manifest(
    paths: list[str],
    participants_tsv: pathlib.Path,
    dataset_name: str,
) -> pd.DataFrame:
    # ---- build dataframe from the paths -----------------------------
    df_paths = pd.DataFrame(
        {
            "image_path": paths,
            "subject_id": [extract_subject(p) for p in paths],
            "modality": [guess_modality(p) for p in paths],
            "dataset": dataset_name,
        }
    )

    # ---- read participants ------------------------------------------
    df_part = pd.read_csv(participants_tsv, sep="\t", dtype=str)
    
    # Check if eligibility column exists
    has_eligibility = 'eligibility' in df_part.columns
    
    # Prepare columns to keep
    columns_to_keep = ["participant_id", "age", "sex"]
    if has_eligibility:
        columns_to_keep.append("eligibility")
    
    # Keep selected columns and rename to match spec
    df_part = df_part[columns_to_keep].rename(
        columns={"participant_id": "subject_id"}
    )

    # ---- merge -------------------------------------------------------
    # Define columns for final dataframe
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset"]
    if has_eligibility:
        result_columns.append("eligibility")
    
    df = (
        df_paths.merge(df_part, on="subject_id", how="left")
        .loc[:, result_columns]
        .sort_values(["subject_id", "modality"])
    )
    return df


# ---------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--participants",
        required=True,
        type=pathlib.Path,
        help="participants.tsv file",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-root",
        type=pathlib.Path,
        help="Root directory that contains sub-*/ses-*/anat/*.nii(.gz) files",
    )
    group.add_argument(
        "--paths-list",
        type=pathlib.Path,
        help="Text file with pre-computed list of image paths (one per line)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="image_manifest.tsv",
        help="Where to write the resulting TSV (default: ./image_manifest.tsv)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nimh_rv",
        help="Dataset name to include as a column (default: nimh_rv)",
    )
    args = parser.parse_args(argv)

    # ---- gather image paths -----------------------------------------
    if args.input_root:
        paths = collect_paths(args.input_root)
    else:
        with open(args.paths_list) as f:
            paths = [line.strip() for line in f if line.strip()]

    if not paths:
        sys.exit("No image paths found!")

    # ---- build manifest ---------------------------------------------
    manifest = build_manifest(paths, args.participants, args.dataset)

    # ---- Create output filenames with dataset name -----------------
    output_path = args.output
    # If output is a file path (not just directory), modify the filename
    if output_path.name:
        # Get file stem and suffix
        stem = output_path.stem
        suffix = output_path.suffix
        # Create new filename with dataset name
        new_filename = f"{stem}_{args.dataset}{suffix}"
        # Update the output path with the new filename
        output_path = output_path.with_name(new_filename)

    # ---- save to specified output path -----------------------------
    manifest.to_csv(output_path, sep="\t", index=False)
    print(f"Manifest written to: {output_path.resolve()}")
    
    # ---- save a copy to script's directory -------------------------
    script_dir = pathlib.Path(__file__).parent
    local_output = script_dir / output_path.name
    manifest.to_csv(local_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {local_output.resolve()}")
    
    print(f"{len(manifest):,} rows")

    # preview first few rows
    print("\n--- preview ---")
    print(manifest.head())


if __name__ == "__main__":
    main()


# python3 nimh_rv_create_dataset.py --participants /mnt/c/Projects/thesis_project/Data/all_demographics/nimh_rv_participants.tsv --input-root /mnt/data/brain_age_preprocessed/NIMH_RV/ --output /mnt/data/brain_age_preprocessed/NIMH_RV/image_manifest.tsv --dataset nimh_rv