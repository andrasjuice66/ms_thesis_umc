"""
Copy all .nii.gz files with T1w in filename (excluding ses-2, ses-3, ses-4) from source to destination,
flattening the directory structure.
"""
import logging
import shutil
from pathlib import Path


def find_nii_files(source_dir):
    """
    Recursively find all .nii.gz files that contain 'T1w' in filename 
    and don't contain excluded sessions/runs in the path.
    """
    log = logging.getLogger()
    log.info(f"Scanning directory for T1w .nii.gz files: {source_dir}")
    
    found_files = []
    excluded_patterns = ["ses-2", "ses-3", "ses-4", "ses-5", "ses-6", "ses-7", "ses-8", "ses-9", "ses-10", "run-2", "run-3", "run-4", "run-5", "run-6", "run-7", "run-8", "run-9", "run-10"]
    
    for file_path in source_dir.rglob("*.nii.gz"):
        path_str = str(file_path)
        has_excluded_pattern = any(pattern in path_str for pattern in excluded_patterns)
        
        if "T1w" in file_path.name and not has_excluded_pattern:
            log.info(f"Found file: {file_path}")
            found_files.append(file_path)
        else:
            if "T1w" not in file_path.name:
                log.debug(f"Skipping (no T1w): {file_path}")
            if has_excluded_pattern:
                excluded_pattern = next(pattern for pattern in excluded_patterns if pattern in path_str)
                log.debug(f"Skipping (contains {excluded_pattern}): {file_path}")
    
    return found_files


def copy_files_flat(source_files, dest_dir):
    """
    Copy files to destination directory without preserving directory structure.
    Handle filename conflicts by adding a counter suffix.
    """
    log = logging.getLogger()
    dest_dir.mkdir(exist_ok=True, parents=True)
    
    copied_count = 0
    skipped_count = 0
    filename_conflicts = {}
    
    for source_file in source_files:
        original_name = source_file.name
        dest_file = dest_dir / original_name
        
        # Handle filename conflicts by adding a counter
        if dest_file.exists():
            if original_name not in filename_conflicts:
                filename_conflicts[original_name] = 1
            else:
                filename_conflicts[original_name] += 1
            
            # Create new filename with counter
            stem = original_name.replace('.nii.gz', '')
            new_name = f"{stem}_{filename_conflicts[original_name]}.nii.gz"
            dest_file = dest_dir / new_name
            log.warning(f"Filename conflict resolved: {original_name} → {new_name}")
        
        try:
            if dest_file.exists():
                log.info(f"Skipping (already exists): {dest_file.name}")
                skipped_count += 1
            else:
                shutil.copy2(source_file, dest_file)
                log.info(f"Copied: {source_file} → {dest_file.name}")
                copied_count += 1
        except Exception as e:
            log.error(f"Failed to copy {source_file}: {e}")
    
    log.info(f"Copy complete: {copied_count} copied, {skipped_count} skipped")
    return copied_count, skipped_count


def main():
    # Hardcoded paths - modify these as needed
    source_dir = Path(r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\data\AOMIC_ID1000")
    dest_dir = Path(r"C:\Projects\thesis_project\Data\AOMIC_ID1000")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    log = logging.getLogger()
    
    log.info(f"Source directory: {source_dir}")
    log.info(f"Destination directory: {dest_dir}")
    
    # Check if source directory exists
    if not source_dir.exists():
        log.error(f"Source directory does not exist: {source_dir}")
        return
    
    # Find all T1w .nii.gz files (excluding ses-2, ses-3, ses-4)
    nii_files = find_nii_files(source_dir)
    total_files = len(nii_files)
    
    if total_files == 0:
        log.warning("No T1w .nii.gz files found (excluding ses-2, ses-3, ses-4 paths)")
        return
    
    log.info(f"Found {total_files} T1w .nii.gz files to copy")
    
    # Copy files with flattened structure
    copied, skipped = copy_files_flat(nii_files, dest_dir)
    
    log.info(f"✅ Process complete!")
    log.info(f"Total T1w files found: {total_files}")
    log.info(f"Files copied: {copied}")
    log.info(f"Files skipped: {skipped}")


if __name__ == "__main__":
    main()