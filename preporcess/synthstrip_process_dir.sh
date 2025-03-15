#!/bin/bash

# Define directories
INPUT_DIR="/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/DataExp/demos/data/images"
OUTPUT_DIR="/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/DataExp/demos/data/images_stripped"
MASK_DIR="/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/DataExp/demos/data/images_masks"

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR" "$MASK_DIR"

# Make sure we have the latest SynthStrip image
docker pull freesurfer/synthstrip

# Process each .nii.gz file in the input directory
for input_file in "$INPUT_DIR"/*.nii.gz; do
    # Get the filename without path
    filename=$(basename "$input_file")
    # Create output filenames
    output_file="${filename%.nii.gz}_stripped.nii.gz"
    mask_file="${filename%.nii.gz}_mask.nii.gz"
    
    echo "Processing $filename..."
    
    # Run SynthStrip directly with Docker
    docker run -v "$INPUT_DIR:/input" -v "$OUTPUT_DIR:/output" -v "$MASK_DIR:/masks" \
      freesurfer/synthstrip \
      -i "/input/$filename" \
      -o "/output/$output_file" \
      -m "/masks/$mask_file"
    
    echo "Completed: $output_file"
done

echo "All files processed successfully!"