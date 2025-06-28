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
conda activate thesis

# 2) navigate to the preprocessing directory
cd $HOME/brain_age_pred/data/preprocess

# 3) execute the conversion script
python convert_nifti_to_npy.py --input /scratch-shared/ajoos/brain_age_preprocessed --output /scratch-shared/ajoos/brain_age_preprocessed_converted --normalize --percentile_low 1 --percentile_high 99 --zero_negatives

#python check_normalistation.py /scratch-shared/ajoos/brain_age_preprocessed_converted -r