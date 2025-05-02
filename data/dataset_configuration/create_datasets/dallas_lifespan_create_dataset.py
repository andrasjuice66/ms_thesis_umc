#!/usr/bin/env python3
"""
Build a manifest for the Dallas Lifespan dataset that links every image to the
participant's demographic information including age at each wave, sex, and race.

Output columns
--------------
subject_id, image_path, sex, age, modality, dataset, wave_nr, race
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


SUB_RE = re.compile(r"(sub-[0-9]+)")
WAVE_RE = re.compile(r"ses-wave([0-9])")


def extract_subject_and_wave(path: str) -> tuple:
    """
    Extract subject ID and wave number from path.
    """
    # Extract subject ID
    subject_match = SUB_RE.search(path)
    if not subject_match:
        raise ValueError(f"No subject id found in: {path}")
    subject_id = subject_match.group(1)
    
    # Extract wave number
    wave_match = WAVE_RE.search(path)
    if not wave_match:
        raise ValueError(f"No wave number found in: {path}")
    wave_nr = wave_match.group(1)
    
    return subject_id, wave_nr


# ---------------------------------------------------------------------
def collect_paths() -> list[str]:
    """
    Return paths for all .nii.gz files in the Dallas Lifespan dataset.
    Hardcoded for simplicity.
    """
    root_dir = pathlib.Path("/mnt/data/brain_age_preprocessed/OpenNeuro/DallasLifeSpan")
    return [str(p) for p in root_dir.rglob("*.nii.gz")]


# ---------------------------------------------------------------------
def build_manifest(paths: list[str]) -> pd.DataFrame:
    """
    Build a manifest dataframe with all required information.
    """
    # Load participants data
    participants_tsv = "/mnt/c/Projects/thesis_project/Data/all_demographics/dallas_participants.tsv"
    df_participants = pd.read_csv(participants_tsv, sep="\t", dtype=str)
    
    # Create paths dataframe with extracted metadata
    path_data = []
    for path in paths:
        subject_id, wave_nr = extract_subject_and_wave(path)
        path_data.append({
            "subject_id": subject_id,
            "image_path": path,
            "modality": guess_modality(path),
            "wave_nr": wave_nr,
        })
    
    df_paths = pd.DataFrame(path_data)
    
    # Prepare participant data with renamed columns
    df_participants = df_participants.rename(columns={
        "participant_id": "subject_id",
        "Sex": "sex",
        "Race": "race",
        "AgeMRI_W1": "age_w1",
        "AgeMRI_W2": "age_w2",
        "AgeMRI_W3": "age_w3"
    })
    
    # Merge paths with participant data
    df_merged = df_paths.merge(
        df_participants[["subject_id", "sex", "race", "age_w1", "age_w2", "age_w3"]], 
        on="subject_id", 
        how="left"
    )
    
    # Create age column based on wave number
    def get_age_for_wave(row):
        wave = row["wave_nr"]
        age_col = f"age_w{wave}"
        if age_col in row and pd.notna(row[age_col]) and row[age_col] != "n/a":
            return row[age_col]
        return "unknown"
    
    df_merged["age"] = df_merged.apply(get_age_for_wave, axis=1)
    
    # Add dataset column
    df_merged["dataset"] = "dallas_lifespan"
    
    # Select and order final columns
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset", "wave_nr", "race"]
    df_final = df_merged[result_columns].sort_values(["subject_id", "wave_nr", "modality"])
    
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
    output_file = script_dir / "image_manifest_dallas_lifespan.tsv"
    
    # Save to file
    manifest.to_csv(output_file, sep="\t", index=False)
    print(f"Manifest written to: {output_file.resolve()}")
    print(f"{len(manifest):,} rows")
    
    # Also save to the data directory
    data_output = pathlib.Path("/mnt/data/OpenNeuro/DallasLifeSpan/image_manifest_dallas_lifespan.tsv")
    manifest.to_csv(data_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {data_output}")
    
    # Preview
    print("\n--- preview ---")
    print(manifest.head())


if __name__ == "__main__":
    main() 