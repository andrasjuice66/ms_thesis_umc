"""
Prediction script for brain age prediction models.
"""

import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm

from configs.config import Config
from data.dataset import BrainAgeDataset
from models.sfcn import SFCN
from models.resnet3d import ResNet3D
from models.efficientnet3d import EfficientNet3D
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict brain age")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output CSV file")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    
    return parser.parse_args()


def predict_single_image(model, image_path, device, transform=None):
    """
    Predict brain age for a single image.
    
    Args:
        model: Model to use
        image_path: Path to input image
        device: Device to use
        transform: Transform to apply to the image
        
    Returns:
        Predicted age
    """
    # Load image
    image = nib.load(image_path)
    image_data = image.get_fdata()
    
    # Apply transform if provided
    if transform is not None:
        data_dict = {"image": image_data}
        data_dict = transform(data_dict)
        image_data = data_dict["image"]
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
    
    # Predict
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
    
    return prediction.item()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = args.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Set up logger
    logger = setup_logger("predict", "predict.log")
    logger.info(f"Arguments: {args}")
    
    # Load model checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model configuration
    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get("type", "sfcn")
    
    # Create model
    logger.info(f"Creating model: {model_type}")
    if model_type == "sfcn":
        model = SFCN(**model_config)
    elif model_type == "resnet3d":
        model = ResNet3D(**model_config)
    elif model_type == "efficientnet3d":
        model = EfficientNet3D(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Get default transform
    transform = BrainAgeDataset._get_default_transform()
    
    # Predict
    input_path = Path(args.input)
    results = []
    
    if input_path.is_file():
        # Single file
        logger.info(f"Predicting for single file: {input_path}")
        prediction = predict_single_image(model, input_path, device, transform)
        results.append({"image_path": str(input_path), "predicted_age": prediction})
    else:
        # Directory
        logger.info(f"Predicting for directory: {input_path}")
        image_files = list(input_path.glob("*.nii.gz"))
        image_files.extend(input_path.glob("*.nii"))
        
        for image_file in tqdm(image_files, desc="Predicting"):
            prediction = predict_single_image(model, image_file, device, transform)
            results.append({"image_path": str(image_file), "predicted_age": prediction})
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    
    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()