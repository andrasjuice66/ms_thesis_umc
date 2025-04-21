import os
import glob
import subprocess
import re
from pathlib import Path
import logging
import argparse
import sys
import shutil
import json
from datetime import datetime
import intensity_normalization
import intensity_normalization.normalize
import intensity_normalization.normalize.nyul
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = 12
import ants
from bids_helper_funcs import create_bids_directories, create_bids_sidecar_json, get_bids_filename, get_bids_modality_dir, create_dataset_description
from transforms import run_n4_bias_correction, register_to_mni, run_brain_extraction, run_denoise_image, run_isotropic_resampling, run_robust_fov, apply_brain_mask, apply_existing_mni_registration, check_and_swap_orientation
from find_files import find_all_input_files, find_all_input_files_bids
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mri_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MRI_Processing")

def process_file(input_file, output_dir, mni_template, t1_mask=None, session_id=None, reference_images_by_session=None, subject_id=None, modality=None):
    """Process a single MRI file through all steps and save in BIDS format"""
    input_path = Path(input_file)
    
    # Extract subject ID and modality from filename if not provided
    if subject_id is None or modality is None:
        filename = input_path.name
        match = re.match(r'(COIM\d+)_([^_]+)_reoriented\.nii\.gz', filename)
        
        if not match:
            logger.warning(f"Filename {filename} does not match expected pattern COIM****_{{modality}}_reoriented.nii.gz")
            if subject_id is None:
                subject_id = input_path.stem.split('_')[0]
            if modality is None:
                # Try to extract modality from filename (without extension)
                modality = os.path.basename(input_file).split('.')[0].lower()
        else:
            if subject_id is None:
                subject_id = match.group(1)
            if modality is None:
                modality = match.group(2)
    
    
    logger.info(f"Processing file: {input_path.name}, Subject: {subject_id}, Modality: {modality}, Session: {session_id}")
    
    # Create BIDS directory structure
    bids_dirs = create_bids_directories(output_dir, subject_id, session_id)
    
    # Create a temporary working directory
    temp_dir = Path(output_dir) / "tmp" / subject_id / f"ses-{session_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    base_name = f"{subject_id}_{modality}_ses-{session_id}"
    t1_name = f"{subject_id}_T1_ses-{session_id}"
    metadata = {
        "OriginalFile": str(input_file),
        "Modality": modality,
        "Session": session_id,
        "ProcessingSteps": []
    }
    
    # Step 1: Check and swap orientation if necessary
    orient_output = temp_dir / f"{base_name}_orient.nii.gz"
    orient_output = check_and_swap_orientation(logger, input_file, orient_output)
    metadata["ProcessingSteps"].append({"Name": "OrientationCheck", "InputFile": str(input_file), "OutputFile": str(orient_output)})
    
    # Step 2: Run robustfov if necessary
    fov_output = temp_dir / f"{base_name}_fov.nii.gz"
    fov_output = run_robust_fov(logger, orient_output, fov_output)
    metadata["ProcessingSteps"].append({"Name": "RobustFOV", "InputFile": str(orient_output), "OutputFile": str(fov_output)})
    
    # Step 3: Voxel isotropic resampling
    iso_output = temp_dir / f"{base_name}_iso.nii.gz"
    iso_output = run_isotropic_resampling(logger, fov_output, iso_output)
    metadata["ProcessingSteps"].append({"Name": "IsotropicResampling", "InputFile": str(fov_output), "OutputFile": str(iso_output)})

    # Step 4: N4 Bias Field Correction
    if modality.lower() in ['ct', 'dose', 'tv_mask']:
        denoise_output = iso_output
    else:
        n4_output = temp_dir / f"{base_name}_n4.nii.gz"
        n4_output = run_n4_bias_correction(logger, iso_output, n4_output)
        metadata["ProcessingSteps"].append({"Name": "N4BiasFieldCorrection", "InputFile": str(iso_output), "OutputFile": str(n4_output)})
        
        # Step 5: Denoise Image
        denoise_output = temp_dir / f"{base_name}_denoise.nii.gz"
        denoise_output = run_denoise_image(logger, n4_output, denoise_output)
        metadata["ProcessingSteps"].append({"Name": "DenoiseImage", "InputFile": str(n4_output), "OutputFile": str(denoise_output)})

    # Step 6: Rigid+affine register to MNI template
    mni_output = temp_dir / f"{base_name}_mni.nii.gz"
    transform_output = temp_dir / f"{t1_name}_transform"
    mni_transform = f"{transform_output}0GenericAffine.mat"
    if modality.lower() in ['t1']:
        mni_output = register_to_mni(logger, denoise_output, mni_output, mni_template, transform_output)
        metadata["ProcessingSteps"].append({"Name": "MNIRegistration", "TemplateFile": str(mni_template), "InputFile": str(denoise_output), "OutputFile": str(mni_output), "TransformOutput": str(transform_output)})
    elif modality.lower() in ['t1gd', 'flair', 'ct', 'tv_mask', 'dose']:
        #mni_output = apply_existing_mni_registration(iso_output, mni_output, mni_transform, temp_dir / f"{t1_name}_mni.nii.gz")
        #metadata["ProcessingSteps"].append({"Name": "ExistingMNIRegistration", "TemplateFile": str(mni_template), "RegistrationFile": str(transform_output), "InputFile": str(iso_output), "OutputFile": str(mni_output)})
        mni_output = register_to_mni(logger, denoise_output, mni_output, temp_dir / f"{t1_name}_mni.nii.gz", temp_dir / f"{base_name}_transform")
        metadata["ProcessingSteps"].append({"Name": "MNIRegistration", "TemplateFile": str(mni_template), "InputFile": str(denoise_output), "OutputFile": str(mni_output), "TransformOutput": str(transform_output)})
    
    # Step 7: Brain extraction / skull stripping (only for T1, T1Gd, or FLAIR modality)
    brain_mask = None
    if t1_mask == None:
        if modality.lower() in ['t1', 't1gd', 'flair']:
            brain_output = temp_dir / f"{base_name}_brain_mask.nii"
            brain_output, mask_output = run_brain_extraction(logger, mni_output, brain_output, mask_file=True)
            brain_mask = mask_output
            metadata["ProcessingSteps"].append({"Name": "BrainExtraction", "InputFile": str(mni_output), "OutputFile": str(brain_output), "MaskFile": str(mask_output)})
            brain_mask = temp_dir / f"{base_name}_brain_mask.nii"
            # Step 8: Multiplication with brain mask
            masked_output = temp_dir / f"{base_name}_masked.nii.gz"
            masked_output = apply_brain_mask(logger, mni_output, brain_mask, masked_output)
            metadata["ProcessingSteps"].append({"Name": "BrainMaskApplication", "InputFile": str(brain_output), "MaskFile": str(mask_output), "OutputFile": str(masked_output)})
            
            # Copy brain mask to BIDS directory
            bids_modality_dir = get_bids_modality_dir(bids_dirs, modality)
            mask_bids_filename = get_bids_filename(subject_id, session_id, modality, "mask")
            mask_bids_output = bids_modality_dir / mask_bids_filename
            shutil.copy(brain_mask, mask_bids_output)
            create_bids_sidecar_json(mask_bids_output, {"SourceFile": str(brain_output), "Description": f"Brain mask generated from {modality} image"})
            
            # Store mask in reference_images_by_session for potential use by other modalities
            if reference_images_by_session is not None:
                if session_id not in reference_images_by_session:
                    reference_images_by_session[session_id] = {}
                reference_images_by_session[session_id]["MASK"] = brain_mask
        
    elif t1_mask:
        # Use T1 mask for other modalities
        masked_output = temp_dir / f"{base_name}_masked.nii.gz"
        masked_output = apply_brain_mask(logger, mni_output, t1_mask, masked_output)
        metadata["ProcessingSteps"].append({"Name": "BrainMaskApplication", "InputFile": str(mni_output), "MaskFile": str(t1_mask), "OutputFile": str(masked_output)})
    else:
        # No masking if T1 mask not available
        masked_output = mni_output
        print("T1 mask not available, using non-masked input")
    
    
    # Copy final output to BIDS directory
    bids_modality_dir = get_bids_modality_dir(bids_dirs, modality)
    bids_filename = get_bids_filename(subject_id, session_id, modality, "skullstripped")
    bids_output = bids_modality_dir / bids_filename
    shutil.copy(masked_output, bids_output)
    
    # Create sidecar JSON
    create_bids_sidecar_json(bids_output, metadata)
    
    logger.info(f"Completed processing for file {input_path.name}")
    return {
        'subject_id': subject_id,
        'modality': modality,
        'session_id': session_id,
        'masked_output': masked_output,
        'mni_output': mni_output,
        'bids_output': bids_output,
        'reference_potential': iso_output,
        'brain_mask': brain_mask
    }

def process_session_files(session_files, subject_id, session_id, output_dir, mni_template, reference_images_by_session):
    """Process all files for a specific session with proper skull stripping hierarchy"""
    processed_results = []
    t1_mask = None
    
    # Check for T1 first
    t1_files = [(f, m) for f, m in session_files if m.lower() == 't1']
    
    # If no T1, check for T1Gd
    if not t1_files:
        t1_files = [(f, m) for f, m in session_files if m.lower() == 't1gd']
    
    # If no T1 or T1Gd, check for FLAIR
    if not t1_files:
        t1_files = [(f, m) for f, m in session_files if m.lower() == 'flair']
    
    # Process the selected file for skull stripping
    if t1_files:
        logger.info(f"Using {t1_files[0][1]} for skull stripping in session {session_id}")
        t1_file, t1_modality = t1_files[0]
        
        result = process_file(
            t1_file,
            output_dir,
            mni_template,
            t1_mask=None,  # No mask yet for this file
            session_id=session_id,
            reference_images_by_session=reference_images_by_session,
            subject_id=subject_id,
            modality=t1_modality
        )
        
        t1_mask = result['brain_mask']
        
        # Store reference image
        if session_id not in reference_images_by_session:
            reference_images_by_session[session_id] = {}
        reference_images_by_session[session_id][t1_modality.upper()] = result['reference_potential']
        if t1_mask:
            reference_images_by_session[session_id]["MASK"] = t1_mask
        
        # Remove processed file from the list
        session_files = [(f, m) for f, m in session_files if f != t1_file]
    else:
        logger.warning(f"No T1, T1Gd, or FLAIR found for subject {subject_id} session {session_id}. Using session 1 mask if available.")
        # Try to use session 1 mask
        if "1" in reference_images_by_session and "MASK" in reference_images_by_session["1"]:
            t1_mask = reference_images_by_session["1"]["MASK"]
    
    # Process remaining files
    for file, modality in session_files:
            result = process_file(
                file,
                output_dir,
                mni_template,
                t1_mask=t1_mask,
                session_id=session_id,
                reference_images_by_session=reference_images_by_session,
                subject_id=subject_id,
                modality=modality
            )
            
            
            # Store reference image for each modality
            if session_id not in reference_images_by_session:
                reference_images_by_session[session_id] = {}
            reference_images_by_session[session_id][modality.upper()] = result['reference_potential']
            
    
    return processed_results, t1_mask

def main():
    parser = argparse.ArgumentParser(description='Process MRI images with various transforms')
    parser.add_argument('--raw_mrs_dir', default="C:/Users/P013630/Documents/data/raw_mrs", help='Directory containing original MRI images')
    parser.add_argument('--fu_nii_std_dir', default="C:/Users/P013630/Documents/data/FU_nii_std", help='Directory containing follow-up MRI images (optional, only for specific dataset)')
    parser.add_argument('--output_dir', default="C:/Users/P013630/Documents/data/processed_mrs_single_test", help='Directory to save processed images')
    parser.add_argument('--mni_template', default="C:/Users/P013630/Documents/data/MNI152lin_T1_1mm.nii.gz", help='Path to MNI template')
    parser.add_argument('--subjects', default="", help='Comma-separated list of subject IDs to process (default: all)')
    parser.add_argument('--modalities', help='Comma-separated list of modalities to process (default: all)')
    parser.add_argument('--threads', type=int, default=12, help='Number of parallel processes')
    parser.add_argument('--del_intermediates', action="store_true", help='Delete temp folder after running')
    parser.add_argument('--bids', action="store_true", help="read from bids or custom file structure")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create temporary directory
    os.makedirs(Path(args.output_dir) / "tmp", exist_ok=True)
    
    # Create BIDS dataset_description.json
    create_dataset_description(args.output_dir)
    
    if args.bids == False:
        # Find all MRI files from both directories
        input_dirs = {
            "raw_mrs": args.raw_mrs_dir,
            "fu_nii_std": args.fu_nii_std_dir
        }
        all_files_with_sessions = find_all_input_files(input_dirs)
    if args.bids == True:
        all_files_with_sessions = find_all_input_files_bids(args.raw_mrs_dir)
    # Filter by subject if specified
    if args.subjects:
        subject_ids = args.subjects.split(',')
        all_files_with_sessions = [(f, s, subj, m) for f, s, subj, m in all_files_with_sessions if subj in subject_ids]
    
    # Filter by modality if specified
    if args.modalities:
        modalities = args.modalities.split(',')
        all_files_with_sessions = [(f, s, subj, m) for f, s, subj, m in all_files_with_sessions if m in modalities]
    
    logger.info(f"Found {len(all_files_with_sessions)} files to process")
    
    # Group files by subject and session
    subject_files = {}
    for file_info in all_files_with_sessions:
        file, session_id, subject_id, modality = file_info
        
        if subject_id not in subject_files:
            subject_files[subject_id] = {}
        
        if session_id not in subject_files[subject_id]:
            subject_files[subject_id][session_id] = []
        
        subject_files[subject_id][session_id].append((file, modality))
    
    # Process each subject's files
    for subject_id, sessions in subject_files.items():
        logger.info(f"Processing subject {subject_id} with {sum(len(files) for files in sessions.values())} files across {len(sessions)} sessions")
        
        # Dictionary to store reference images by session and modality
        reference_images_by_session = {}
        
        # Now process sessions in order
        for session_id in sorted([s for s in sessions.keys()]):
            logger.info(f"Processing session {session_id} for subject {subject_id}")
            session_files = sessions[session_id]
            
            # Process session files with skull stripping hierarchy
            _ = process_session_files(
                session_files, 
                subject_id, 
                session_id, 
                args.output_dir, 
                args.mni_template, 
                reference_images_by_session
            )
        
       
    
    logger.info("All processing completed")
    
    # Clean up temporary directory
    if args.del_intermediates == True:
        shutil.rmtree(Path(args.output_dir) / "tmp")

if __name__ == "__main__":
    main()

