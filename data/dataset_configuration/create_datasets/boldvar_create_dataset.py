#!/usr/bin/env python3
"""
Build a manifest for the BOLD Variability dataset that links every image to the
participant's demographic information.

Output columns
--------------
subject_id, image_path, sex, age, modality, dataset, performance
"""
import pathlib
import re
import sys
import os
import pandas as pd


# ---------------------------------------------------------------------
def guess_modality(path: str) -> str:
    """
    Return modality based on the filename.
    """
    fname = pathlib.Path(path).name.lower()
    if "_t1w" in fname:
        return "t1"
    if "_t2w" in fname:
        return "t2"
    if "flair" in fname:
        return "flair"
    return "unknown"


SUB_RE = re.compile(r"(sub-[0-9]+)")


def extract_subject_id(path: str) -> str:
    """
    Extract subject ID from the path.
    """
    # Extract subject ID from filename
    filename = pathlib.Path(path).name
    subject_match = SUB_RE.search(filename)
    if not subject_match:
        raise ValueError(f"No subject id found in: {path}")
    subject_id = subject_match.group(1)
    
    return subject_id


# ---------------------------------------------------------------------
def collect_paths() -> list[str]:
    """
    Return paths for all .nii.gz files in the BOLD Variability dataset.
    Hardcoded for simplicity.
    """
    # The files are directly in this directory, not in sub-X/anat/ subdirectories
    root_dir = pathlib.Path("/mnt/data/brain_age_preprocessed/OpenNeuro/BoldVariability")
    
    # Check if directory exists
    if not root_dir.exists():
        print(f"Warning: Directory does not exist: {root_dir}")
        return []
    
    # Get all .nii.gz files directly from the root directory
    paths = [str(p) for p in root_dir.glob("sub-*_*.nii.gz")]
    
    # If we found files, return them
    if paths:
        return paths
    
    # If no files found, print directory contents for debugging
    print("No files found with glob pattern. Directory contents:")
    try:
        contents = list(root_dir.iterdir())
        for item in contents[:20]:  # Show first 20 items to avoid too much output
            print(f"  {item.name}")
        if len(contents) > 20:
            print(f"  ... and {len(contents) - 20} more items")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    return []


# ---------------------------------------------------------------------
def build_manifest(paths: list[str]) -> pd.DataFrame:
    """
    Build a manifest dataframe with all required information.
    """
    # Load participants data
    participants_tsv = "/mnt/c/Projects/thesis_project/Data/all_demographics/bold_var_participants.tsv"
    df_participants = pd.read_csv(participants_tsv, sep="\t", dtype=str)
    
    # Create paths dataframe with extracted metadata
    path_data = []
    for path in paths:
        try:
            subject_id = extract_subject_id(path)
            path_data.append({
                "subject_id": subject_id,
                "image_path": path,
                "modality": guess_modality(path),
            })
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    df_paths = pd.DataFrame(path_data)
    
    # Add performance column if it exists in participants data
    if "perf" in df_participants.columns:
        df_participants["performance"] = df_participants["perf"]
    
    # Select columns to include from participants data
    participant_columns = ["participant_id", "age", "sex"]
    if "performance" in df_participants.columns:
        participant_columns.append("performance")
    
    # Merge paths with participant data
    df_merged = df_paths.merge(
        df_participants[participant_columns],
        left_on="subject_id",
        right_on="participant_id",
        how="left"
    )
    
    # Add dataset column
    df_merged["dataset"] = "boldvar"
    
    # Select and order final columns
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset"]
    if "performance" in df_merged.columns:
        result_columns.append("performance")
    
    # Filter to only include columns that exist
    result_columns = [col for col in result_columns if col in df_merged.columns]
    
    df_final = df_merged[result_columns].sort_values(["subject_id", "modality"])
    
    return df_final


# ---------------------------------------------------------------------
def main():
    # Collect paths
    print("Collecting image paths...")
    paths = collect_paths()
    
    if not paths:
        sys.exit("No image paths found!")
    
    print(f"Found {len(paths)} images")
    
    # Build manifest
    print("Building manifest...")
    manifest = build_manifest(paths)
    
    # Create output path with dataset name
    script_dir = pathlib.Path(__file__).parent
    output_file = script_dir / "image_manifest_boldvar.tsv"
    
    # Save to file
    manifest.to_csv(output_file, sep="\t", index=False)
    print(f"Manifest written to: {output_file.resolve()}")
    print(f"{len(manifest):,} rows")
    
    # Also save to the data directory
    data_dir = pathlib.Path("/mnt/data/brain_age_preprocessed/OpenNeuro/BoldVariability")
    data_output = data_dir / "image_manifest_boldvar.tsv"
    manifest.to_csv(data_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {data_output}")
    
    # Preview
    print("\n--- preview ---")
    print(manifest.head())


if __name__ == "__main__":
    main() 