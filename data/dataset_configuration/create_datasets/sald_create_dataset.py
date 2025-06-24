#!/usr/bin/env python3
"""
Build a manifest for the SALD dataset that links every image to the
participant's demographic information.

Output columns
--------------
subject_id, image_path, sex, age, modality, dataset
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
    Return paths for all .nii.gz files in the SALD dataset.
    """
    # The files are in BIDS format: sub-ID/anat/sub-ID_T1w.nii.gz
    root_dir = pathlib.Path("/mnt/c/Projects/thesis_project/Data/SALD/SALD_AWS_SYNC/RawData_BIDS")
    
    # Check if directory exists
    if not root_dir.exists():
        print(f"Warning: Directory does not exist: {root_dir}")
        return []
    
    # Get all .nii.gz files in BIDS structure
    paths = []
    for pattern in ["sub-*/anat/sub-*_T1w.nii.gz"]:
        paths.extend([str(p) for p in root_dir.glob(pattern)])
    
    # Remove duplicates and sort
    paths = sorted(list(set(paths)))
    
    # If we found files, return them
    if paths:
        print(f"Found {len(paths)} image files")
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
    participants_tsv = "/mnt/c/Projects/thesis_project/brain_age_pred/data/dataset_configuration/all_demographics/sald_participants.tsv"
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
    
    # Clean demographics data - SALD has duplicate Sex columns, use the first one
    # Rename columns to match expected format
    df_participants_clean = df_participants.rename(columns={
        'Sub_ID': 'participant_id',
        'Age': 'age',
        'Sex': 'sex'  # Use the first Sex column
    })
    
    # Convert subject ID format: from numeric (031274) to sub-031274
    df_participants_clean['participant_id'] = 'sub-' + df_participants_clean['participant_id'].astype(str)
    
    # Select columns to include from participants data
    participant_columns = ["participant_id", "age", "sex"]
    
    # Filter out rows with missing age or sex data
    df_participants_clean = df_participants_clean[participant_columns].dropna()
    
    # Merge paths with participant data
    df_merged = df_paths.merge(
        df_participants_clean,
        left_on="subject_id",
        right_on="participant_id",
        how="inner"  # Only keep subjects with both image and demographic data
    )
    
    # Add dataset column
    df_merged["dataset"] = "sald"
    
    # Select and order final columns
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset"]
    
    # Filter to only include columns that exist
    result_columns = [col for col in result_columns if col in df_merged.columns]
    
    df_final = df_merged[result_columns].sort_values(["subject_id", "modality"])
    
    # Ensure each subject appears only once by taking the first occurrence
    df_final = df_final.drop_duplicates(subset=['subject_id'], keep='first')
    
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
    output_file = script_dir / "image_manifest_sald.tsv"
    
    # Save to file
    manifest.to_csv(output_file, sep="\t", index=False)
    print(f"Manifest written to: {output_file.resolve()}")
    print(f"{len(manifest):,} rows")
    
    # Also save to the data directory
    data_dir = pathlib.Path("/mnt/c/Projects/thesis_project/Data/SALD")
    data_output = data_dir / "image_manifest_sald.tsv"
    manifest.to_csv(data_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {data_output}")
    
    # Preview
    print("\n--- preview ---")
    print(manifest.head())
    
    # Show some statistics
    print(f"\n--- statistics ---")
    print(f"Total subjects: {len(manifest)}")
    print(f"Sex distribution:")
    print(manifest['sex'].value_counts())
    print(f"Age range: {manifest['age'].astype(float).min():.1f} - {manifest['age'].astype(float).max():.1f}")
    print(f"Modality distribution:")
    print(manifest['modality'].value_counts())


if __name__ == "__main__":
    main() 