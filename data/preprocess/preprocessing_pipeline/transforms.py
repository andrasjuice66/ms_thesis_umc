
from pathlib import Path
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = 4
import ants
import nibabel as nib
from wsl_helper_funcs import run_wsl_command, windows_to_wsl_path

def run_n4_bias_correction(logger, input_file, output_file):
    """Run N4BiasFieldCorrection using ANTsPy"""
    logger.info(f"Running N4BiasFieldCorrection on {input_file}")
    print(input_file)
    # Load the image using ANTsPy
    img = ants.image_read(str(input_file))
    
    # Apply N4 bias field correction
    corrected_img = ants.n4_bias_field_correction(img)
    
    # Save the corrected image
    ants.image_write(corrected_img, str(output_file))
    
    logger.info(f"N4BiasFieldCorrection completed: {output_file}")
    return output_file

def run_denoise_image(logger, input_file, output_file):
    """Run DenoiseImage using ANTsPy"""
    logger.info(f"Running DenoiseImage on {input_file}")
    
    import ants
    # Load the image using ANTsPy
    img = ants.image_read(str(input_file))
    
    # Apply Rician denoising
    denoised_img = ants.denoise_image(img, noise_model='Rician')
    
    # Save the denoised image
    ants.image_write(denoised_img, str(output_file))
    
    logger.info(f"DenoiseImage completed: {output_file}")
    return output_file

def check_and_swap_orientation(logger, input_file, output_file=None):
    """Check and swap FSL orientation if necessary"""
    if output_file is None:
        output_file = input_file
        
    wsl_input = windows_to_wsl_path(input_file)
    wsl_output = windows_to_wsl_path(output_file)
    
    # Check current orientation
    cmd = f"\"fslorient -getorient {wsl_input}\""
    orientation = run_wsl_command(logger, cmd)
    
    # If needed, swap orientation
    if orientation == "NEUROLOGICAL":
        logger.info(f"Swapping orientation for {input_file}")
        if input_file != output_file:
            # Copy file first
            cmd = f"cp {wsl_input} {wsl_output}"
            run_wsl_command(logger, cmd, use_fsl=False)
        
        cmd = f"\"fslorient -swaporient {wsl_output}\""
        run_wsl_command(logger, cmd)
        logger.info(f"Orientation swapped: {output_file}")
    else:
        if input_file != output_file:
            # Just copy the file if no swap needed
            cmd = f"cp {wsl_input} {wsl_output}"
            run_wsl_command(logger, cmd, use_fsl=False)
            logger.info(f"No orientation swap needed, copied: {output_file}")
        else:
            logger.info(f"No orientation swap needed: {output_file}")
    
    return output_file

def run_robust_fov(logger, input_file, output_file):
    """Run robustfov to restrict field of view to brain only"""
    wsl_input = windows_to_wsl_path(input_file)
    wsl_output = windows_to_wsl_path(output_file)
    
    cmd = f"\"robustfov -i {wsl_input} -r {wsl_output}\""
    run_wsl_command(logger, cmd)
    logger.info(f"robustfov completed: {output_file}")
    return output_file

def run_isotropic_resampling(logger, input_file, output_file):
    """Resample to isotropic voxels using flirt"""
    wsl_input = windows_to_wsl_path(input_file)
    wsl_output = windows_to_wsl_path(output_file)
    
    cmd = f"\"flirt -in {wsl_input} -ref {wsl_input} -applyisoxfm 1 -nosearch -interp trilinear -out {wsl_output}\""
    run_wsl_command(logger, cmd)
    logger.info(f"Isotropic resampling completed: {output_file}")
    return output_file

def run_brain_extraction(logger, input_file, output_file, mask_file=None):
    """Run brain extraction / skull stripping using FSL BET"""
    wsl_input = windows_to_wsl_path(input_file)
    wsl_output_prefix = windows_to_wsl_path(output_file.parent / output_file.stem)
    
    if mask_file:
        cmd = f"\"export FREESURFER_HOME=/usr/local/freesurfer/8.0.0 && source /usr/local/freesurfer/8.0.0/SetUpFreeSurfer.sh && /usr/local/freesurfer/8.0.0/bin/mri_synthstrip -i {wsl_input} -m {wsl_output_prefix}.nii\""
        run_wsl_command(logger, cmd, use_fsl=True)
        logger.info(f"Brain extraction completed: {output_file}")
        # The mask will be saved as {output_prefix}_mask.nii.gz
        return output_file, Path(f"{output_file.parent}/{output_file.stem}.gz")
    else:
        cmd = f"bet {wsl_input} {wsl_output_prefix}"
        run_wsl_command(logger, cmd, use_fsl=True)
        logger.info(f"Brain extraction completed: {output_file}")
        return output_file

def apply_brain_mask(logger, input_file, mask_file, output_file):
    """Apply brain mask using nib"""
    try:
        # Load the input image and mask
        img = nib.load(input_file)
        mask = nib.load(mask_file)

        # Get the data arrays
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()

        # Multiply the image data with the mask
        masked_data = img_data * mask_data

        # Create a new NIfTI image with the masked data
        # Use the header and affine from the original image
        masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)

        # Save the masked image
        nib.save(masked_img, output_file)

        logger.info(f"Brain mask applied: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error applying brain mask: {str(e)}")
        raise

def register_to_mni(logger, input_file, output_file, mni_template, transform_output):
    """Register to MNI template using ANTsPy and save transform matrices

    Args:
        input_file (str or Path): Path to the input image
        output_file (str or Path): Path for the registered output image
        mni_template (str or Path): Path to the MNI template

    Returns:
        tuple: (output_file path, forward_transform path, inverse_transform path)
    """
    logger.info(f"Registering {input_file} to MNI template")

    # Load the images using ANTsPy
    fixed = ants.image_read(str(mni_template))
    moving = ants.image_read(str(input_file))


    # Run registration with rigid + affine transforms
    reg_result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='Affine',
        initial_transform=None,
        outprefix=str(transform_output)
    )

    # Save the registered image
    ants.image_write(reg_result['warpedmovout'], str(output_file))

    # The transform files are automatically saved with these extensions:
    # Forward transform (moving to fixed): _transform0GenericAffine.mat
    forward_transform = Path(f"{str(transform_output)}0GenericAffine.mat")

    logger.info(f"Registration to MNI completed: {output_file}")
    logger.info(f"Forward transform saved: {forward_transform}")

    return output_file

def apply_existing_mni_registration(logger, input_file, output_file, transform_file, mni_template):
    """Apply existing MNI registration transform using ANTsPy

    Args:
        input_file (str or Path): Path to the input image to be transformed
        output_file (str or Path): Path where the transformed image will be saved
        transform_file (str or Path): Path to the existing transform file (.mat for affine)

    Returns:
        Path: Path to the transformed output file
    """
    logger.info(f"Applying existing MNI registration transform to {input_file}")

    # Load the input image
    fixed = ants.image_read(str(mni_template))
    moving = ants.image_read(str(input_file))

    # Apply the existing transform
    transformed = ants.apply_transforms(
        fixed=fixed,  # Not needed when just applying transform
        moving=moving,
        transformlist=[str(transform_file)],
        interpolator='linear'
    )

    # Save the transformed image
    ants.image_write(transformed, str(output_file))

    logger.info(f"Transform application completed: {output_file}")
    return output_file