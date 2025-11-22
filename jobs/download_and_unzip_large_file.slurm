#!/bin/bash
#SBATCH --job-name=brain_age_download
#SBATCH --output=brain_age_%j.out
#SBATCH --error=brain_age_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


# Set up directories
SCRATCH_DIR="/scratchshared/ajoos"
mkdir -p ${SCRATCH_DIR}

# Navigate to scratch directory
cd ${SCRATCH_DIR}

echo "Starting download at $(date)"

# Download brain_age_preprocessed_no_csf.zip
echo "Downloading brain_age_preprocessed_no_csf.zip..."
curl -L -o brain_age_preprocessed_no_csf.zip "https://surfdrive.surf.nl/index.php/s/40kI877ttDfKyEy/download?path=%2F&files=brain_age_preprocessed_no_csf.zip"

if [ $? -eq 0 ]; then
    echo "Successfully downloaded brain_age_preprocessed_no_csf.zip"
else
    echo "Failed to download brain_age_preprocessed_no_csf.zip"
    exit 1
fi

# Download brain_age_segmented.zip
echo "Downloading brain_age_segmented.zip..."
curl -L -o brain_age_segmented.zip "https://surfdrive.surf.nl/index.php/s/40kI877ttDfKyEy/download?path=%2F&files=brain_age_segmented.zip"

if [ $? -eq 0 ]; then
    echo "Successfully downloaded brain_age_segmented.zip"
else
    echo "Failed to download brain_age_segmented.zip"
    exit 1
fi

echo "Downloads completed at $(date)"

# Unzip files
echo "Starting unzip at $(date)"

echo "Unzipping brain_age_preprocessed_no_csf.zip..."
unzip -q brain_age_preprocessed_no_csf.zip
if [ $? -eq 0 ]; then
    echo "Successfully unzipped brain_age_preprocessed_no_csf.zip"
    rm brain_age_preprocessed_no_csf.zip  # Optional: remove zip after extraction
else
    echo "Failed to unzip brain_age_preprocessed_no_csf.zip"
fi

echo "Unzipping brain_age_segmented.zip..."
unzip -q brain_age_segmented.zip
if [ $? -eq 0 ]; then
    echo "Successfully unzipped brain_age_segmented.zip"
    rm brain_age_segmented.zip  # Optional: remove zip after extraction
else
    echo "Failed to unzip brain_age_segmented.zip"
fi

echo "All operations completed at $(date)"

# List contents to verify
echo "Contents of ${SCRATCH_DIR}:"
ls -lh ${SCRATCH_DIR}