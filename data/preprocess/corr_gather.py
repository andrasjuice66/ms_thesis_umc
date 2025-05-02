import os
import shutil
import argparse
from pathlib import Path


def gather_tsv_files(input_dir, output_dir):
    """
    Find all .tsv files in input_dir and its subdirectories,
    rename them with parent folder as prefix, and copy to output_dir.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track success and failure counts
    success_count = 0
    error_count = 0
    
    # Walk through all directories and files in input_dir
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tsv'):
                # Get parent directory name
                parent_dir = os.path.basename(root)
                
                # Create new filename with parent directory as prefix
                new_filename = f"{parent_dir}_{file}"
                
                # Source and destination paths
                source_path = os.path.join(root, file)
                dest_path = os.path.join(output_dir, new_filename)
                
                # Copy the file with error handling
                try:
                    shutil.copy(source_path, dest_path)  # Using copy instead of copy2 to avoid permission issues with metadata
                    print(f"Copied: {source_path} -> {dest_path}")
                    success_count += 1
                except PermissionError:
                    print(f"Error: Permission denied when copying {source_path}")
                    error_count += 1
                except Exception as e:
                    print(f"Error copying {source_path}: {str(e)}")
                    error_count += 1
    
    print(f"\nSummary: {success_count} files copied successfully, {error_count} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather TSV files with parent directory as filename prefix")
    parser.add_argument("input_dir", help="Directory to search for TSV files")
    parser.add_argument("output_dir", help="Directory to output renamed TSV files")
    
    args = parser.parse_args()
    
    gather_tsv_files(args.input_dir, args.output_dir)
