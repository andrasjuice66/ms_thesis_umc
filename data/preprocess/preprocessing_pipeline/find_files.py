import os
import glob
import re
from pathlib import Path

def find_all_input_files(input_dirs):
    """Find all input files across multiple directories"""
    all_files = []
    
    # Process files from raw_mrs directory (session 1)
    if "raw_mrs" in input_dirs:
        raw_mrs_files = glob.glob(os.path.join(input_dirs["raw_mrs"], "COIM*_*_reoriented.nii.gz"))
        for f in raw_mrs_files:
            # Extract subject ID and modality from filename
            filename = os.path.basename(f)
            match = re.match(r'(COIM\d+)_([^_]+)_reoriented\.nii\.gz', filename)
            if match:
                subject_id = match.group(1)
                modality = match.group(2)
                all_files.append((f, "1", subject_id, modality))  # Session 1
    
    # Process files from FU_nii_std directory (follow-up sessions)
    if "fu_nii_std" in input_dirs:
        fu_base_dir = input_dirs["fu_nii_std"]
        # Find all COIM**** directories
        coim_dirs = glob.glob(os.path.join(fu_base_dir, "COIM*"))
        
        for coim_dir in coim_dirs:
            subject_id = os.path.basename(coim_dir)
            # Find all FU* directories
            fu_dirs = glob.glob(os.path.join(coim_dir, "FU*"))
            
            for fu_dir in fu_dirs:
                # Extract FU number
                fu_match = re.search(r'FU(\d+)', os.path.basename(fu_dir))
                if fu_match:
                    fu_num = int(fu_match.group(1))
                    session_id = str(fu_num + 1)  # FU1 corresponds to ses-2
                    
                    # Find all nii.gz files in this FU directory
                    fu_files = glob.glob(os.path.join(fu_dir, "*.nii.gz"))
                    for f in fu_files:
                        # Extract modality from filename (without extension)
                        modality = os.path.basename(f).split('.')[0].lower()
                        if modality == "flair":
                            modality = "FLAIR"
                        if modality == "t1":
                            modality = "T1"
                        if modality == "t1gd":
                            modality = "T1Gd"
                        # Create a tuple with file path, session ID, subject ID, and modality
                        all_files.append((f, session_id, subject_id, modality))
    
    return all_files

def find_all_input_files_bids(bids_root):
    """Find all input files in a BIDS directory structure

    Args:
        bids_root (str): Path to the root of the BIDS directory

    Returns:
        list of tuples: (file_path, session_id, subject_id, modality)
    """
    all_files = []
    bids_root = Path(bids_root)

    # Find all subject directories
    subject_dirs = list(bids_root.glob("sub-*"))

    for subject_dir in subject_dirs:
        # Extract subject ID from directory name
        subject_id = subject_dir.name.replace('sub-', '')

        # Find all session directories for this subject
        session_dirs = list(subject_dir.glob("ses-*"))
        if not session_dirs:  # If no sessions, assume single session in subject dir
            session_dirs = [subject_dir]

        for session_dir in session_dirs:
            # Extract session ID from directory name, default to "1" if no session specified
            if 'ses-' in session_dir.name:
                session_id = session_dir.name.replace('ses-', '')
            else:
                session_id = "1"

            # Look for anat directory which typically contains structural images
            anat_dir = session_dir / 'anat'
            if anat_dir.exists():
                # Find all NIfTI files
                nifti_files = list(anat_dir.glob("*.nii.gz"))

                for nifti_file in nifti_files:
                    # Extract modality from filename following BIDS convention
                    # Example: sub-001_ses-01_T1w.nii.gz
                    modality_match = re.search(r'_([A-Za-z1]+w)\.nii\.gz$', nifti_file.name)
                    if modality_match:
                        modality = modality_match.group(1)
                        # Map BIDS modalities to your expected format
                        modality_mapping = {
                            'T1w': 'T1',
                            'T1Gd': 'T1Gd',  # Assuming this naming convention
                            'FLAIR': 'FLAIR'
                        }
                        modality = modality_mapping.get(modality, modality)

                        all_files.append((
                            str(nifti_file),
                            session_id,
                            subject_id,
                            modality
                        ))

    return all_files