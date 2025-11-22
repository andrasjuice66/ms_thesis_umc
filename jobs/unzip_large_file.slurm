#!/bin/bash
#SBATCH --job-name=unzip_brain_age_data
#SBATCH --partition=gpu_a100            
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8           # Minimal CPU allocation
#SBATCH --mem=4G                           # Minimal memory allocation
#SBATCH --time=01:30:00                    # 6 hours for both files
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END    
#SBATCH --mail-user=a.g.joos@vu.student.nl

set -x
set -o pipefail

# Define destination directory
DEST_DIR="/scratch-shared/ajoos/"

# Change to destination directory
cd "$DEST_DIR"

echo "Starting unzip jobs at: $(date)"

# Unzip first file
ZIP_FILE1="/scratch-shared/ajoos/brain_age_preprocessed_no_csf.zip"
echo "Starting to unzip: $ZIP_FILE1"
echo "Start time: $(date)"

unzip -o "$ZIP_FILE1"

if [ $? -eq 0 ]; then
    echo "Successfully unzipped: $ZIP_FILE1"
    echo "Disk usage:"
    du -sh brain_age_preprocessed_no_csf/
else
    echo "Error: Failed to unzip $ZIP_FILE1"
    exit 1
fi

echo "First file completed at: $(date)"

# Unzip second file
ZIP_FILE2="/scratch-shared/ajoos/brain_age_segmented.zip"
echo "Starting to unzip: $ZIP_FILE2"
echo "Start time: $(date)"

unzip -o "$ZIP_FILE2"

if [ $? -eq 0 ]; then
    echo "Successfully unzipped: $ZIP_FILE2"
    echo "Disk usage:"
    du -sh brain_age_segmented/
else
    echo "Error: Failed to unzip $ZIP_FILE2"
    exit 1
fi

echo "All unzip jobs completed at: $(date)"
echo "Total disk usage of both extracted folders:"
du -sh brain_age_preprocessed_no_csf/ brain_age_segmented/
