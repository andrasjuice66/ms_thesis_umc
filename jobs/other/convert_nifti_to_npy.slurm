#!/bin/bash
#SBATCH --job-name=convert_nifti
#SBATCH --partition=gpu_a100        
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END    
#SBATCH --mail-user=a.g.joos@student.vu.nl

set -x
set -o pipefail

# 1) activate your env
source ~/.bashrc
conda activate brain_age_310

# 2) navigate to the preprocessing directory
cd $HOME/brain_age_pred/data/preprocess

# 3) execute the conversion script WITHOUT normalization
python convert_nifti_to_npy.py --input /scratch-shared/ajoos1/brain_age_segmented/ --output /scratch-shared/ajoos1/brain_age_segmented_converted/ --no-normalize

#python check_normalistation.py /scratch-shared/ajoos1/brain_age_preprocessed_converted -r