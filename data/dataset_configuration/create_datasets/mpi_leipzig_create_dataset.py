#!/usr/bin/env python3
"""
Build a manifest for the MPI-Leipzig dataset that links every image to the
participant's demographic information.

Output columns
--------------
subject_id, image_path, sex, age, modality, dataset, session_id, acquisition
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


def extract_acquisition(path: str) -> str:
    """
    Extract acquisition info from path if available.
    """
    fname = pathlib.Path(path).name.lower()
    acq_match = re.search(r"acq-([a-zA-Z0-9]+)", fname)
    if acq_match:
        return acq_match.group(1)
    return "standard"


SUB_RE = re.compile(r"(sub-[0-9]+)")
SESS_RE = re.compile(r"(ses-[0-9]+)")


def extract_metadata(path: str) -> tuple:
    """
    Extract subject ID and session ID from the path.
    """
    path_str = str(path)
    
    # Extract subject ID
    subject_match = SUB_RE.search(path_str)
    if not subject_match:
        raise ValueError(f"No subject id found in: {path}")
    subject_id = subject_match.group(1)
    
    # Extract session ID
    session_match = SESS_RE.search(path_str)
    if not session_match:
        session_id = "ses-01"  # Default if not found
    else:
        session_id = session_match.group(1)
    
    return subject_id, session_id


# ---------------------------------------------------------------------
def collect_paths() -> list[str]:
    """
    Return paths for all .nii.gz files in the MPI-Leipzig dataset.
    Hardcoded for simplicity.
    """
    root_dir = pathlib.Path("/mnt/data/brain_age_preprocessed/OpenNeuro/MPI_Leipzig")
    return [str(p) for p in root_dir.rglob("*.nii.gz")]


# ---------------------------------------------------------------------
def process_age_bins(age_bin: str) -> float:
    """
    Process age bins like '25-30' to return the mean (27.5).
    """
    if pd.isna(age_bin) or not age_bin:
        return None
    
    try:
        # Extract the two numbers from the bin
        lower, upper = map(int, age_bin.split('-'))
        # Return the mean
        return (lower + upper) / 2
    except (ValueError, AttributeError):
        print(f"Warning: Could not process age bin: {age_bin}")
        return None


# ---------------------------------------------------------------------
def build_manifest(paths: list[str]) -> pd.DataFrame:
    """
    Build a manifest dataframe with all required information.
    """
    # Load participants data
    participants_tsv = "/mnt/c/Projects/thesis_project/Data/all_demographics/mpi_leipzig_participants.tsv"
    df_participants = pd.read_csv(participants_tsv, sep="\t", dtype=str)
    
    # Process age bins to get mean ages
    if "age (5-year bins)" in df_participants.columns:
        df_participants["age"] = df_participants["age (5-year bins)"].apply(process_age_bins)
    
    # Standardize gender column
    if "gender" in df_participants.columns:
        df_participants["sex"] = df_participants["gender"].apply(
            lambda x: "m" if x.upper() == "M" else "f" if x.upper() == "F" else "unknown"
        )
    
    # Create paths dataframe with extracted metadata
    path_data = []
    for path in paths:
        try:
            subject_id, session_id = extract_metadata(path)
            acquisition = extract_acquisition(path)
            path_data.append({
                "subject_id": subject_id,
                "session_id": session_id,
                "acquisition": acquisition,
                "image_path": path,
                "modality": guess_modality(path),
            })
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    df_paths = pd.DataFrame(path_data)
    
    # Merge paths with participant data
    df_merged = df_paths.merge(
        df_participants[["participant_id", "age", "sex"]],
        left_on="subject_id",
        right_on="participant_id",
        how="left"
    )
    
    # Add dataset column
    df_merged["dataset"] = "mpi_leipzig"
    
    # Select and order final columns
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset", "session_id", "acquisition"]
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
    output_file = script_dir / "image_manifest_mpi_leipzig.tsv"
    
    # Save to file
    manifest.to_csv(output_file, sep="\t", index=False)
    print(f"Manifest written to: {output_file.resolve()}")
    print(f"{len(manifest):,} rows")
    
    # Also save to the data directory
    data_output = pathlib.Path("/mnt/data/OpenNeuro/MPI_Leipzig/image_manifest_mpi_leipzig.tsv")
    manifest.to_csv(data_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {data_output}")
    
    # Preview
    print("\n--- preview ---")
    print(manifest.head())


if __name__ == "__main__":
    main() 