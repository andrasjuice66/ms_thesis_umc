#!/usr/bin/env python3
"""
Build a manifest for the CamCAN dataset that links every image to the
participant's demographic information including age, gender, and handedness.

Output columns
--------------
subject_id, image_path, sex, age, modality, dataset, handedness, tiv
"""
import pathlib
import re
import sys
import os
import pandas as pd


# ---------------------------------------------------------------------
def guess_modality(path: str) -> str:
    """
    Return 't1' or 't2' based on the filename.
    """
    fname = pathlib.Path(path).name.lower()
    if "_t1w" in fname:
        return "t1"
    if "_t2w" in fname:
        return "t2"
    return "unknown"


SUB_RE = re.compile(r"(sub-CC[0-9]+)")


def extract_subject(path: str) -> str:
    """
    Extract subject ID from path.
    """
    # Extract subject ID
    subject_match = SUB_RE.search(path)
    if not subject_match:
        raise ValueError(f"No subject id found in: {path}")
    subject_id = subject_match.group(1)
    
    return subject_id


# ---------------------------------------------------------------------
def collect_paths() -> list[str]:
    """
    Return paths for all .nii.gz files in the CamCAN dataset.
    Hardcoded for simplicity.
    """
    root_dir = pathlib.Path("/mnt/data/brain_age_preprocessed/CamCAN")
    return [str(p) for p in root_dir.rglob("*.nii.gz")]


# ---------------------------------------------------------------------
def build_manifest(paths: list[str]) -> pd.DataFrame:
    """
    Build a manifest dataframe with all required information.
    """
    # Load participants data
    participants_tsv = "/mnt/c/Projects/thesis_project/Data/all_demographics/camcan_participants.tsv"
    df_participants = pd.read_csv(participants_tsv, sep="\t", dtype=str)
    
    # Create paths dataframe with extracted metadata
    path_data = []
    for path in paths:
        subject_id = extract_subject(path)
        path_data.append({
            "subject_id": subject_id,
            "image_path": path,
            "modality": guess_modality(path),
        })
    
    df_paths = pd.DataFrame(path_data)
    
    # Prepare participant data with renamed columns
    df_participants = df_participants.rename(columns={
        "participant_id": "subject_id",
        "gender_code": "sex_code",
        "hand": "handedness",
        "tiv_cubicmm": "tiv"
    })
    
    # Map gender_code to standardized sex format
    def map_sex(row):
        code = row.get("sex_code")
        if code == "1":
            return "m"
        elif code == "2":
            return "f"
        return "unknown"
    
    df_participants["sex"] = df_participants.apply(map_sex, axis=1)
    
    # Merge paths with participant data
    df_merged = df_paths.merge(
        df_participants[["subject_id", "sex", "age", "handedness", "tiv"]], 
        on="subject_id", 
        how="left"
    )
    
    # Add dataset column
    df_merged["dataset"] = "camcan"
    
    # Select and order final columns
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset", "handedness", "tiv"]
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
    output_file = script_dir / "image_manifest_camcan.tsv"
    
    # Save to file
    manifest.to_csv(output_file, sep="\t", index=False)
    print(f"Manifest written to: {output_file.resolve()}")
    print(f"{len(manifest):,} rows")
    
    # Also save to the data directory
    data_output = pathlib.Path("/mnt/data/brain_age_preprocessed/CamCAN/image_manifest_camcan.tsv")
    manifest.to_csv(data_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {data_output}")
    
    # Preview
    print("\n--- preview ---")
    print(manifest.head())


if __name__ == "__main__":
    main() 