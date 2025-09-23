#!/usr/bin/env python3
# UNA inference script

import os
import sys
import argparse
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
    model_cfg = os.path.join(una_repo_path, 'cfgs/trainer/test/test.yaml')
    gen_cfg = os.path.join(una_repo_path, 'cfgs/generator/test/test.yaml')
    
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
    
    # Create flipped version (simplified approach)
    img_flip = torch.flip(img, dims=[0])
    flip_path = os.path.join(output_dir, f"{base_name}_flip_reg2orig.nii.gz")
    viewVolume(img_flip, aff, names=[f"{base_name}_flip_reg2orig"], save_dir=output_dir)
    img_flip_reg2orig_path = os.path.join(output_dir, f"{base_name}_flip_reg2orig.nii.gz")
    
    # Read the flipped image back (to match UNA's pipeline)
    _, img_flip_reg2orig, _, _ = utils.prepare_image(img_flip_reg2orig_path, win_size=win_size, spacing=None, im_only=True, device=device)
    
    # Run inference
    print("Running UNA inference...")
    outs = utils.evaluate_image(img, img_flip_reg2orig, ckp_path=model_path, device=device, gen_cfg=gen_cfg, model_cfg=model_cfg)
    
    # Save results
    for k, v in outs.items():
        if isinstance(v, torch.Tensor):
            viewVolume(v, aff, names=[f"out_{k}"], save_dir=output_dir)
    
    print(f"Results saved to: {output_dir}")
    return outs

def main():
    parser = argparse.ArgumentParser(description="Run UNA inference on a brain MRI")
    parser.add_argument("--input", type=str, required=True, help="Path to input image (.nii.gz)")
    parser.add_argument("--output", type=str, default="./una_results", help="Output directory")
    parser.add_argument("--model", type=str, default="./una.pth", help="Path to UNA model weights (una.pth)")
    parser.add_argument("--win_size", type=int, nargs=3, default=[160, 160, 160], help="Window size for processing")
    
    args = parser.parse_args()
    
    run_una_inference(args.input, args.output, args.model, args.win_size)

if __name__ == "__main__":
    main()