#!/usr/bin/env python3
# UNA inference script
import os
import sys
import torch
from pathlib import Path

# Add UNA to path
una_repo_path = os.path.dirname(os.path.abspath(__file__)) + "/OtherRepos/UNA"
sys.path.append(una_repo_path)

# Import UNA utilities
import utils.test_utils as utils
from utils.misc import viewVolume, make_dir

def run_una_inference(input_path, output_dir, model_path, win_size=[160, 160, 160]):
    """
    Run UNA inference on a single image
    
    Args:
        input_path (str): Path to the input image (.nii.gz)
        output_dir (str): Directory to save results
        model_path (str): Path to una.pth weights
        win_size (list): Window size for processing
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set up paths
    model_cfg = os.path.join(una_repo_path, '/mnt/c/Projects/thesis_project/brain_age_pred/UNA/cfgs/trainer/test/test.yaml')
    gen_cfg = os.path.join(una_repo_path, '/mnt/c/Projects/thesis_project/brain_age_pred/UNA/cfgs/generator/test/test.yaml')
    
    # Create output directory
    output_dir = make_dir(output_dir, reset=False)
    
    print(f"Processing image: {input_path}")
    
    # Create flipped and registered version of input image
    # This is required by UNA's pipeline
    input_dir = os.path.dirname(input_path)
    input_name = os.path.basename(input_path)
    base_name = input_name.split('.nii')[0]
    
    # Read the original image
    _, img, _, aff = utils.prepare_image(input_path, win_size=win_size, im_only=True, device=device)
    
    # Create flipped version - save to output directory temporarily
    img_flip = torch.flip(img, dims=[0])
    flip_name = f"{base_name}_flip_reg2orig"
    viewVolume(img_flip, aff, names=[flip_name], save_dir=output_dir)
    img_flip_reg2orig_path = os.path.join(output_dir, f"{flip_name}.nii.gz")
    
    # Read the flipped image back (to match UNA's pipeline)
    _, img_flip_reg2orig, _, _ = utils.prepare_image(img_flip_reg2orig_path, win_size=win_size, spacing=None, im_only=True, device=device)
    
    # Run inference
    print("Running UNA inference...")
    outs = utils.evaluate_image(img, img_flip_reg2orig, ckp_path=model_path, device=device, gen_cfg=gen_cfg, model_cfg=model_cfg)
    
    # Save only the T1 result with the original filename
    if 'T1' in outs and isinstance(outs['T1'], torch.Tensor):
        viewVolume(outs['T1'], aff, names=[base_name], save_dir=output_dir)
        print(f"T1 conversion saved as: {os.path.join(output_dir, input_name)}")
    else:
        print("Warning: No T1 output found in results")
        print(f"Available outputs: {list(outs.keys())}")
    
    # Clean up the temporary flip file
    if os.path.exists(img_flip_reg2orig_path):
        os.remove(img_flip_reg2orig_path)
        print(f"Cleaned up temporary file: {img_flip_reg2orig_path}")
    
    return outs

if __name__ == "__main__":
    # Hardcoded values - modify these as needed
    input_path = "/mnt/c/Projects/thesis_project/Data/brain_age_preprocessed_no_csf/IXI/IXI012-HH-1211-T2.nii.gz"  

    output_dir = "/mnt/c/Projects/thesis_project/Data/una_results/IXI/"         # Output directory
    model_path = "/mnt/c/Projects/thesis_project/brain_age_pred/UNA/assets/una-001.pth"            # Path to UNA model weights
    win_size = [160, 160, 160]          # Window size for processing
    
    run_una_inference(input_path, output_dir, model_path, win_size)