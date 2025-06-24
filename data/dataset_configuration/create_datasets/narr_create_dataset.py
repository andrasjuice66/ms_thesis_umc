#!/usr/bin/env python3
"""
Build a manifest for the Narratives dataset that links every image to the
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
    """Extract subject ID from file path."""
    match = SUB_RE.search(path)
    if match:
        return match.group(1)
    return None


def parse_age_field(age_str: str) -> float:
    """
    Parse age field which can be comma-separated for multiple sessions.
    Return the first valid age found.
    """
    if pd.isna(age_str) or age_str == "n/a":
        return None
        
    # Handle comma-separated ages
    ages = str(age_str).split(',')
    for age in ages:
        age = age.strip()
        if age and age != "n/a":
            try:
                return float(age)
            except ValueError:
                continue
    return None


def clean_sex(sex_str: str) -> str:
    """
    Clean sex field which can be comma-separated for multiple sessions.
    Return the first valid sex found.
    """
    if pd.isna(sex_str) or sex_str == "n/a":
        return None
        
    # Handle comma-separated sex values
    sexes = str(sex_str).split(',')
    for sex in sexes:
        sex = sex.strip().upper()
        if sex in ['M', 'F']:
            return sex
    return None


# ---------------------------------------------------------------------
def main():
    # Paths
    demographics_path = "/mnt/c/Projects/thesis_project/brain_age_pred/data/dataset_configuration/create_datasets/narr_participants.tsv"
    image_dir = "/mnt/c/Projects/thesis_project/Data/brain_age_segmented/OpenNeuro/Narrative"
    output_path = "/mnt/c/Projects/thesis_project/brain_age_pred/data/dataset_configuration/image_manifest_narr.tsv"
    
    # Check if demographics file exists
    if not os.path.exists(demographics_path):
        print(f"Demographics file not found: {demographics_path}")
        sys.exit(1)
    
    # Load demographics
    print(f"Loading demographics from {demographics_path}")
    try:
        df_demo = pd.read_csv(demographics_path, sep='\t')
        print(f"Loaded {len(df_demo)} demographic records")
        print(f"Columns: {list(df_demo.columns)}")
    except Exception as e:
        print(f"Error loading demographics: {e}")
        sys.exit(1)
    
    # Find all image files
    print(f"Scanning for images in {image_dir}")
    image_files = []
    
    if os.path.exists(image_dir):
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.nii.gz', '.nii')):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
    
    print(f"Found {len(image_files)} image files")
    
    # Build manifest
    manifest_rows = []
    subjects_with_images = set()
    subjects_without_demo = set()
    
    for image_path in image_files:
        subject_id = extract_subject_id(image_path)
        if not subject_id:
            print(f"Could not extract subject ID from: {image_path}")
            continue
            
        subjects_with_images.add(subject_id)
        
        # Find demographic info
        demo_row = df_demo[df_demo['participant_id'] == subject_id]
        
        if demo_row.empty:
            subjects_without_demo.add(subject_id)
            continue
        
        # Get first matching demographic record
        demo = demo_row.iloc[0]
        
        # Parse age and sex (handle comma-separated values)
        age = parse_age_field(demo['age'])
        sex = clean_sex(demo['sex'])
        
        # Skip if missing critical demographic info
        if age is None or sex is None:
            print(f"Skipping {subject_id}: missing age ({age}) or sex ({sex})")
            continue
        
        # Determine modality
        modality = guess_modality(image_path)
        
        manifest_rows.append({
            'subject_id': subject_id,
            'image_path': image_path,
            'sex': sex,
            'age': age,
            'modality': modality,
            'dataset': 'narratives'
        })
    
    # Create manifest DataFrame
    if manifest_rows:
        df_manifest = pd.DataFrame(manifest_rows)
        
        # Remove duplicates (keep first occurrence per subject)
        print(f"Before deduplication: {len(df_manifest)} rows")
        df_manifest = df_manifest.drop_duplicates(subset=['subject_id'], keep='first')
        print(f"After deduplication: {len(df_manifest)} rows")
        
        # Sort by subject_id
        df_manifest = df_manifest.sort_values('subject_id')
        
        # Save manifest
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_manifest.to_csv(output_path, sep='\t', index=False)
        print(f"Saved manifest to {output_path}")
        
        # Summary statistics
        print("\n=== SUMMARY ===")
        print(f"Total subjects with images: {len(subjects_with_images)}")
        print(f"Subjects without demographics: {len(subjects_without_demo)}")
        print(f"Final manifest entries: {len(df_manifest)}")
        print(f"Age range: {df_manifest['age'].min():.1f} - {df_manifest['age'].max():.1f}")
        print(f"Sex distribution: {df_manifest['sex'].value_counts().to_dict()}")
        print(f"Modality distribution: {df_manifest['modality'].value_counts().to_dict()}")
        
        if subjects_without_demo:
            print(f"\nSubjects with images but no demographics:")
            for subj in sorted(subjects_without_demo):
                print(f"  {subj}")
    
    else:
        print("No valid manifest entries created!")
        sys.exit(1)


if __name__ == "__main__":
    main() 