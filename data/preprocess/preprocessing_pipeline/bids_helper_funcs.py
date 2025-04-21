import os
from pathlib import Path
import json
from datetime import datetime

def create_bids_directories(output_dir, subject_id, session_id="01"):
    """Create BIDS-compliant directory structure"""
    # Create subject directory
    subject_dir = Path(output_dir) / f"sub-{subject_id}"
    os.makedirs(subject_dir, exist_ok=True)
    
    # Create session directory
    session_dir = subject_dir / f"ses-{session_id}"
    os.makedirs(session_dir, exist_ok=True)
    
    # Create modality directories
    anat_dir = session_dir / "anat"
    os.makedirs(anat_dir, exist_ok=True)
    
    func_dir = session_dir / "func"
    os.makedirs(func_dir, exist_ok=True)
    
    dwi_dir = session_dir / "dwi"
    os.makedirs(dwi_dir, exist_ok=True)
    
    ct_dir = session_dir / "ct"
    os.makedirs(ct_dir, exist_ok=True)
    
    other_dir = session_dir / "other"
    os.makedirs(other_dir, exist_ok=True)
    
    return {
        'subject': subject_dir,
        'session': session_dir,
        'anat': anat_dir,
        'func': func_dir,
        'dwi': dwi_dir,
        'ct': ct_dir,
        'other': other_dir
    }

def get_bids_modality_dir(bids_dirs, modality):
    """Get the appropriate BIDS directory for a given modality"""
    modality_lower = modality.lower()
    
    if modality_lower in ['t1', 't2', 't1w', 't2w', 'flair', 'pd', 't1gd']:
        return bids_dirs['anat']
    elif modality_lower in ['bold', 'cbv', 'cbf', 'asl']:
        return bids_dirs['func']
    elif modality_lower in ['dwi', 'dti']:
        return bids_dirs['dwi']
    elif modality_lower in ['ct']:
        return bids_dirs['ct']
    else:
        return bids_dirs['other']

def get_bids_filename(subject_id, session_id, modality, suffix, extension=".nii.gz"):
    """Generate a BIDS-compliant filename"""
    modality_lower = modality.lower()
    
    # Map modality to BIDS entity
    if modality_lower == 't1':
        modality_entity = 'T1w'
    elif modality_lower == 't1gd':
        modality_entity = 'T1w_gd'
    elif modality_lower == 't2':
        modality_entity = 'T2w'
    elif modality_lower == 'flair':
        modality_entity = 'FLAIR'
    elif modality_lower == 'pd':
        modality_entity = 'PD'
    elif modality_lower == 'bold':
        modality_entity = 'bold'
    elif modality_lower == 'dwi':
        modality_entity = 'dwi'
    elif modality_lower == 'ct':
        modality_entity = 'CT'
    else:
        modality_entity = modality_lower
    
    # Construct BIDS filename
    if suffix:
        filename = f"sub-{subject_id}_ses-{session_id}_{modality_entity}_{suffix}{extension}"
    else:
        filename = f"sub-{subject_id}_ses-{session_id}_{modality_entity}{extension}"
    
    return filename

def create_bids_sidecar_json(output_file, metadata):
    """Create a BIDS sidecar JSON file"""
    json_file = str(output_file).replace('.nii.gz', '.json')
    
    # Add basic metadata
    metadata.update({
        "GeneratedBy": {
            "Name": "MRI Processing Pipeline",
            "Version": "1.0",
            "DateCreated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }
    })
    
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return json_file

def create_dataset_description(output_dir):
    """Create a BIDS dataset_description.json file"""
    description = {
        "Name": "Processed MRI Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
        "License": "CC0",
        "Authors": ["Your Name"],
        "Acknowledgements": "Thanks to the MRI processing pipeline",
        "HowToAcknowledge": "Please cite this dataset when using it",
        "GeneratedBy": [{
            "Name": "MRI Processing Pipeline",
            "Version": "1.0",
            "CodeURL": "https://github.com/yourusername/mri-processing"
        }],
        "DataProcessing": {
            "Pipeline": "Custom MRI processing pipeline with ANTsPy and FSL"
        }
    }
    
    with open(Path(output_dir) / "dataset_description.json", 'w') as f:
        json.dump(description, f, indent=4)