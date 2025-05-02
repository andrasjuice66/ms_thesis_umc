#!/usr/bin/env python3
"""
Build a manifest for the NPC (Neuroscience and Psychology of Creativity) dataset
that links every image to the participant's demographic information.

Output columns
--------------
subject_id, image_path, sex, age, modality, dataset, creativity_index
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
    return "unknown"


SUB_RE = re.compile(r"(sub-[0-9]+)")


def extract_subject_id(path: str) -> str:
    """
    Extract subject ID from the path.
    """
    # Extract subject ID
    subject_match = SUB_RE.search(str(path))
    if not subject_match:
        raise ValueError(f"No subject id found in: {path}")
    subject_id = subject_match.group(1)
    
    return subject_id


# ---------------------------------------------------------------------
def collect_paths() -> list[str]:
    """
    Return paths for all .nii.gz files in the NPC dataset.
    Hardcoded for simplicity.
    """
    root_dir = pathlib.Path("/mnt/data/brain_age_preprocessed/OpenNeuro/NPC")
    return [str(p) for p in root_dir.rglob("*.nii.gz")]


# ---------------------------------------------------------------------
def build_manifest(paths: list[str]) -> pd.DataFrame:
    """
    Build a manifest dataframe with all required information.
    """
    # Load participants data
    participants_tsv = "/mnt/c/Projects/thesis_project/Data/all_demographics/npc_participants.tsv"
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
    
    # Standardize sex column (convert 'male'/'female' to 'm'/'f')
    def standardize_sex(sex_value):
        if pd.isna(sex_value) or sex_value == "n/a":
            return "unknown"
        sex_value = str(sex_value).lower()
        if sex_value.startswith('m'):
            return "m"
        elif sex_value.startswith('f'):
            return "f"
        return "unknown"
    
    df_participants["standardized_sex"] = df_participants["sex"].apply(standardize_sex)
    
    # Add creativity index calculated from CBI_Overall (if available)
    if "CBI_Overall" in df_participants.columns:
        df_participants["creativity_index"] = df_participants["CBI_Overall"]
    else:
        df_participants["creativity_index"] = "unknown"
    
    # Select columns to include in final manifest
    participant_columns = ["participant_id", "age", "standardized_sex", "creativity_index"]
    
    # Rename columns to match expected format
    column_mapping = {
        "participant_id": "subject_id",
        "standardized_sex": "sex"
    }
    
    # Merge paths with participant data
    df_merged = df_paths.merge(
        df_participants[participant_columns],
        left_on="subject_id",
        right_on="participant_id",
        how="left"
    )
    
    # Rename columns after merge
    for old_col, new_col in column_mapping.items():
        if old_col in df_merged.columns:
            df_merged[new_col] = df_merged[old_col]
    
    # Add dataset column
    df_merged["dataset"] = "npc"
    
    # Select and order final columns
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset", "creativity_index"]
    
    # Filter to include only columns that exist
    available_columns = [col for col in result_columns if col in df_merged.columns or col in ["subject_id", "image_path", "modality", "dataset"]]
    
    df_final = df_merged[available_columns].sort_values(["subject_id", "modality"])
    
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
    output_file = script_dir / "image_manifest_npc.tsv"
    
    # Save to file
    manifest.to_csv(output_file, sep="\t", index=False)
    print(f"Manifest written to: {output_file.resolve()}")
    print(f"{len(manifest):,} rows")
    
    # Also save to the data directory
    data_output = pathlib.Path("/mnt/data/OpenNeuro/NPC/image_manifest_npc.tsv")
    manifest.to_csv(data_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {data_output}")
    
    # Preview
    print("\n--- preview ---")
    print(manifest.head())


if __name__ == "__main__":
    main() 