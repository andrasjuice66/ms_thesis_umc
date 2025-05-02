#!/usr/bin/env python3
"""
Build a manifest for the OASIS3 dataset that links every image to the
participant's demographic information.

Output columns
--------------
subject_id, image_path, sex, age, modality, dataset, session_id, race, education
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


SUB_RE = re.compile(r"(sub-OAS3[0-9]+)")
SESS_RE = re.compile(r"ses[s]*-(d[0-9]+)")


def extract_metadata(path: str) -> tuple:
    """
    Extract subject ID and session ID from the path.
    """
    # Extract subject ID
    subject_match = SUB_RE.search(path)
    if not subject_match:
        raise ValueError(f"No subject id found in: {path}")
    subject_id = subject_match.group(1)
    
    # Extract session ID
    session_match = SESS_RE.search(path)
    if not session_match:
        raise ValueError(f"No session id found in: {path}")
    session_id = session_match.group(1)
    
    # OAS3 ID without the 'sub-' prefix for demographics matching
    oas_id = subject_id[4:]  # Remove 'sub-' prefix
    
    return subject_id, session_id, oas_id


# ---------------------------------------------------------------------
def collect_paths() -> list[str]:
    """
    Return paths for all .nii.gz files in the OASIS3 dataset.
    Hardcoded for simplicity.
    """
    root_dir = pathlib.Path("/mnt/data/brain_age_preprocessed/OASIS/OASIS3")
    return [str(p) for p in root_dir.rglob("*.nii.gz")]


# ---------------------------------------------------------------------
def build_manifest(paths: list[str]) -> pd.DataFrame:
    """
    Build a manifest dataframe with all required information.
    """
    # Load participants data
    demographics_csv = "/mnt/c/Projects/thesis_project/Data/all_demographics/oasis3_demographics.csv"
    df_demographics = pd.read_csv(demographics_csv, dtype=str)
    
    # Create paths dataframe with extracted metadata
    path_data = []
    for path in paths:
        try:
            subject_id, session_id, oas_id = extract_metadata(path)
            path_data.append({
                "subject_id": subject_id,
                "oas_id": oas_id,
                "session_id": session_id,
                "image_path": path,
                "modality": guess_modality(path),
            })
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    df_paths = pd.DataFrame(path_data)
    
    # Map gender (1=M, 2=F) to standardized format
    def map_sex(row):
        gender = row.get("GENDER")
        if gender == "1":
            return "m"
        elif gender == "2":
            return "f"
        return "unknown"
    
    # Prepare demographics data with proper column names from your actual data
    df_demographics["sex"] = df_demographics.apply(map_sex, axis=1)
    
    # Rename columns to match expected format
    demographics_columns = {
        "OASISID": "oas_id",
        "AgeatEntry": "age",
        "EDUC": "education",
        "race": "race"
    }
    
    # Merge paths with demographic data based on OAS ID
    df_merged = df_paths.merge(
        df_demographics[list(demographics_columns.keys()) + ["sex"]], 
        left_on="oas_id",
        right_on="OASISID", 
        how="left"
    )
    
    # Rename the columns after merging
    for old_col, new_col in demographics_columns.items():
        if old_col in df_merged.columns:
            df_merged[new_col] = df_merged[old_col]
    
    # Add dataset column
    df_merged["dataset"] = "oasis3"
    
    # Select and order final columns
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset", "session_id", "race", "education"]
    
    # Filter to only include columns that exist
    result_columns = [col for col in result_columns if col in df_merged.columns or col in ["subject_id", "image_path", "modality", "dataset", "session_id"]]
    
    df_final = df_merged[result_columns].sort_values(["subject_id", "session_id", "modality"])
    
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
    output_file = script_dir / "image_manifest_oasis3.tsv"
    
    # Save to file
    manifest.to_csv(output_file, sep="\t", index=False)
    print(f"Manifest written to: {output_file.resolve()}")
    print(f"{len(manifest):,} rows")
    
    # Also save to the data directory
    data_output = pathlib.Path("/mnt/data/OASIS/OASIS3/image_manifest_oasis3.tsv")
    manifest.to_csv(data_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {data_output}")
    
    # Preview
    print("\n--- preview ---")
    print(manifest.head())


if __name__ == "__main__":
    main() 