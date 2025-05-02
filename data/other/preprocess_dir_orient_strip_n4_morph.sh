#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#######  EDIT THESE  THREE  VARS  ##########################
# 1) Base of all your datasets (after WSL mount of Z:)
BASE_ROOT="/mnt/neuroRT/data"

# 2) Name of the dataset folder you want to process (e.g. IXI, OASIS,…)
DATASET="IXI"

# 3) Relative path to MNI template under BASE_ROOT
MNI_ROOT="/mnt/c/Projects/thesis_project/ms_thesis_umc/data/preprocess/MNI152_T1_1mm_Brain.nii.gz"
#############################################################

# Derived paths
DATA_ROOT="$BASE_ROOT/$DATASET"
MNI_IMG="$MNI_ROOT"
OUT_ROOT="$BASE_ROOT/brain_age_preprocessed"
OUT_DIR="$OUT_ROOT/$DATASET"

mkdir -p "$OUT_DIR"

####  FUNCTION: check and swap orientation if 'NEUROLOGICAL'  ####
check_and_swap_orientation() {
  local input_file="$1"
  local output_file="${2:-$1}"

  # Ensure output directory exists
  mkdir -p "$(dirname "$output_file")"

  # Get current orientation
  local orient
  orient=$(fslorient -getorient "$input_file")

  if [[ "$orient" == "NEUROLOGICAL" ]]; then
    echo " ↪ Swapping orientation for: $input_file"
    if [[ "$input_file" != "$output_file" ]]; then
      cp "$input_file" "$output_file"
    fi
    fslorient -swaporient "$output_file"
    echo "   → Swapped saved to: $output_file"
  else
    if [[ "$input_file" != "$output_file" ]]; then
      echo " ↪ No swap needed; copying to: $output_file"
      cp "$input_file" "$output_file"
    else
      echo " ↪ No swap needed: $input_file"
    fi
  fi
}

# Find ALL T1/T2/FLAIR .nii(.gz) images (case‑insensitive)
mapfile -t imgs < <(find "$DATA_ROOT" -type f \( \
    -iname "*t1*.nii*" \
 -o -iname "*t2*.nii*" \
 -o -iname "*flair*.nii*" \
\) )

echo "Found ${#imgs[@]} images in $DATA_ROOT"

for img in "${imgs[@]}"; do
  echo
  echo "Input: $img"
  base=$(basename "$img" .nii.gz)
  base=${base%.nii}

  # 1) Orientation check & swap
  oriented="$OUT_DIR/${base}_oriented.nii.gz"
  echo " → Orientation check → $oriented"
  check_and_swap_orientation "$img" "$oriented"

  # 2) N4 bias‑field correction
  bias_out="$OUT_DIR/${base}_n4.nii.gz"
  echo " → N4BiasFieldCorrection → $bias_out"
  docker run --rm \
    -v "$BASE_ROOT":/mnt \
    antsx/ants N4BiasFieldCorrection \
      -d 3 \
      -i "/mnt${oriented#$BASE_ROOT}" \
      -o "/mnt${bias_out#$BASE_ROOT}"

  # 3) Skull‑strip
  brain_out="$OUT_DIR/${base}_brain.nii.gz"
  echo " → SynthStrip → $brain_out"
  docker run --rm \
    -v "$BASE_ROOT":/mnt \
    freesurfer/synthstrip \
      -i "/mnt${bias_out#$BASE_ROOT}" \
      -o "/mnt${brain_out#$BASE_ROOT}"

  # 4) Affine register brain → MNI
  reg_out="$OUT_DIR/${base}_aff2mni.nii.gz"
  xfm_out="$OUT_DIR/${base}_aff2mni.lta"
  echo " → SynthMorph → $reg_out + $xfm_out"
  docker run --rm \
    -v "$BASE_ROOT":/mnt \
    freesurfer/synthmorph \
      register -m affine \
        -o "/mnt${reg_out#$BASE_ROOT}" \
        -t "/mnt${xfm_out#$BASE_ROOT}" \
        "/mnt${brain_out#$BASE_ROOT}" \
        "/mnt${MNI_IMG#$BASE_ROOT}"
done

echo
echo "✅ All done. Outputs in: $OUT_DIR"