#!/usr/bin/env python3
"""
Build a manifest for the IXI dataset that links every image to the
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
    if "-t1." in fname or "-t1-" in fname:
        return "t1"
    if "-t2." in fname or "-t2-" in fname:
        return "t2"
    return "unknown"


# The IXI ID format is like "IXI002-Guys-0828" or "IXI013-HH-1212"
ID_RE = re.compile(r"(IXI[0-9]+-[A-Za-z]+-[0-9]+)")


def extract_subject_id(path: str) -> str:
    """
    Extract subject ID from the path.
    """
    path_name = pathlib.Path(path).name
    id_match = ID_RE.search(path_name)
    
    if not id_match:
        # Try extracting from the path itself if not in filename
        id_match = ID_RE.search(str(path))
    
    if not id_match:
        # If still not found, try simpler pattern just with IXI number
        simple_id = re.search(r"(IXI[0-9]+)", str(path))
        if simple_id:
            return simple_id.group(1)
        raise ValueError(f"No subject id found in: {path}")
    
    return id_match.group(1)


# ---------------------------------------------------------------------
def collect_paths() -> list[str]:
    """
    Return paths for all .nii.gz files in the IXI dataset.
    Hardcoded for simplicity.
    """
    root_dir = pathlib.Path("/mnt/data/brain_age_preprocessed/IXI")
    return [str(p) for p in root_dir.rglob("*.nii.gz")]


# ---------------------------------------------------------------------
def build_manifest(paths: list[str]) -> pd.DataFrame:
    """
    Build a manifest dataframe with all required information.
    """
    # Load demographics data from XLS file
    demographics_file = "/mnt/c/Projects/thesis_project/Data/all_demographics/ixi.xls"
    
    try:
        # Try with pandas read_excel
        df_demographics = pd.read_excel(demographics_file)  # Remove dtype=str to keep numeric types
    except Exception as e:
        print(f"Error reading .xls file with pandas: {e}")
        print("Trying with xlrd...")
        try:
            # Fallback to xlrd if needed
            import xlrd
            workbook = xlrd.open_workbook(demographics_file)
            sheet = workbook.sheet_by_index(0)
            
            # Get headers from first row
            headers = [sheet.cell_value(0, col) for col in range(sheet.ncols)]
            
            # Build data list
            data = []
            for row in range(1, sheet.nrows):
                row_data = {}
                for col in range(sheet.ncols):
                    row_data[headers[col]] = sheet.cell_value(row, col)  # Keep original type
                data.append(row_data)
            
            df_demographics = pd.DataFrame(data)
        except Exception as ex:
            print(f"Error reading with xlrd too: {ex}")
            print("Please install required dependencies: pip install xlrd openpyxl")
            sys.exit("Failed to read demographics file")
    
    # Print available columns for debugging
    print("Available columns in demographics data:", df_demographics.columns.tolist())
    
    # Create paths dataframe with extracted metadata
    path_data = []
    for path in paths:
        try:
            subject_id = extract_subject_id(path)
            # Extract IXI number for matching with demographics
            ixi_number = int(re.search(r"IXI([0-9]+)", subject_id).group(1))
            
            path_data.append({
                "subject_id": subject_id,
                "ixi_number": ixi_number,
                "image_path": path,
                "modality": guess_modality(path),
            })
        except (ValueError, AttributeError) as e:
            print(f"Warning: {e}")
            continue
    
    df_paths = pd.DataFrame(path_data)
    
    # Map sex (1=M, 2=F) to standardized format
    def map_sex(sex_value):
        if sex_value == 1:
            return "m"
        elif sex_value == 2:
            return "f"
        return "unknown"
    
    # Create proper sex column
    sex_column = "SEX_ID (1=m, 2=f)"
    if sex_column in df_demographics.columns:
        df_demographics["sex"] = df_demographics[sex_column].apply(map_sex)
    else:
        print(f"Warning: {sex_column} column not found in demographics file")
        df_demographics["sex"] = "unknown"
    
    # Handle IXI_ID mapping for joining
    if "IXI_ID" in df_demographics.columns:
        # Make sure the IXI_ID column is treated as integer
        df_demographics["ixi_number"] = df_demographics["IXI_ID"].astype(int)
        print(f"Using IXI_ID column for matching")
    else:
        print("Warning: IXI_ID column not found in demographics file")
        df_demographics["ixi_number"] = None
    
    # Get age from the demographics
    age_column = "AGE"
    if age_column in df_demographics.columns:
        df_demographics["age"] = df_demographics[age_column]
    else:
        print(f"Warning: {age_column} column not found in demographics file")
        df_demographics["age"] = None
    
    # Print a sample of data for debugging
    print("\nSample of paths data:")
    print(df_paths.head())
    print("\nSample of demographics data:")
    print(df_demographics[["ixi_number", "sex", "age"]].head())
    
    # Merge paths with demographic data - use inner join to check mappings
    print(f"Paths shape before merge: {df_paths.shape}")
    print(f"Demographics shape before merge: {df_demographics.shape}")
    
    df_merged = df_paths.merge(
        df_demographics[["ixi_number", "sex", "age"]], 
        on="ixi_number", 
        how="left"
    )
    
    print(f"Merged shape: {df_merged.shape}")
    print(f"NaN counts in merged data: \n{df_merged[['sex', 'age']].isna().sum()}")
    
    # Add dataset column
    df_merged["dataset"] = "ixi"
    
    # Select and order final columns
    result_columns = ["subject_id", "image_path", "sex", "age", "modality", "dataset"]
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
    output_file = script_dir / "image_manifest_ixi.tsv"
    
    # Save to file
    manifest.to_csv(output_file, sep="\t", index=False)
    print(f"Manifest written to: {output_file.resolve()}")
    print(f"{len(manifest):,} rows")
    
    # Also save to the data directory
    data_output = pathlib.Path("/mnt/data/brain_age_preprocessed/IXI/image_manifest_ixi.tsv")
    manifest.to_csv(data_output, sep="\t", index=False)
    print(f"Copy of manifest written to: {data_output}")
    
    # Preview
    print("\n--- preview ---")
    print(manifest.head())


if __name__ == "__main__":
    main() 