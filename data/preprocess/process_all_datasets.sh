#!/bin/bash

# Base directory
BASE_DIR="/mnt/data"

# Python script path
SCRIPT_PATH="/mnt/c/Projects/thesis_project/ms_thesis_umc/data/preprocess/preprocess_orient_fov_strip_morph.py"

# Datasets to process
DATASETS=(
  # "CamCAN"
  # "OpenNeuro/BoldVariability"
  # "OpenNeuro/DallasLifeSpan"
  "OpenNeuro/MPI_Leipzig"
  # "OpenNeuro/Narrative"
  # "OpenNeuro/NPC"
  # "OASIS/OASIS3"
  # "NIMH_RV"
  # "CoRR"
)

# Process each dataset
for dataset in "${DATASETS[@]}"; do
  echo "Processing dataset: $dataset"
  python3 "$SCRIPT_PATH" "$BASE_DIR" "$dataset"
  echo "Completed processing: $dataset"
  echo "----------------------------------------"
done

echo "All datasets processed successfully!"