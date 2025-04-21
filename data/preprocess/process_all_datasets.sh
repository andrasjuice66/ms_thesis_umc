#!/bin/bash

# Base directory
BASE_DIR="/mnt/neuroRT"

# Python script path
SCRIPT_PATH="/mnt/c/Projects/thesis_project/ms_thesis_umc/data/preprocess/preprocess_dir_orient_strip_n4_morph.py"

# Datasets to process
DATASETS=(
  "CamCAN"
  "CoRR"
  "OASIS/OASIS3"
  "OpenNeuro/BoldVariability"
  "OpenNeuro/DallasLifeSpan"
  "OpenNeuro/MRI-Leipzig"
  "OpenNeuro/Narrative"
  "OpenNeuro/NPC"
)

# Process each dataset
for dataset in "${DATASETS[@]}"; do
  echo "Processing dataset: $dataset"
  python3 "$SCRIPT_PATH" "$BASE_DIR" "$dataset"
  echo "Completed processing: $dataset"
  echo "----------------------------------------"
done

echo "All datasets processed successfully!"