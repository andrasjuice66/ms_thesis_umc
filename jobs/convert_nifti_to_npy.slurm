#!/bin/bash
#SBATCH --job-name=convert_nifti
#SBATCH --partition=tcn rome            
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
python convert_nifti_to_npy.py --input /scratch-shared/ajoos/brain_age_segmented --output /scratch-shared/ajoos/brain_age_segmented_converted 