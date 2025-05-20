#!/bin/bash
#SBATCH --job-name=brainagenext
#SBATCH --partition=gpu_h100            
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/brain_age_pred/output/run_logs/%x_%j.out   
#SBATCH --mail-type=FAIL,END               

set -euo pipefail

# 1) activate your env
source ~/.bashrc
conda activate thesis

# 2) (optional) copy large data to fast local SSD on the node
# SCRATCH=/scratch/$USER/$SLURM_JOB_ID
# mkdir -p $SCRATCH && cp -r /project/<proj>/data/dataset $SCRATCH

# 3) launch training exactly as on your laptop
cd $HOME/
python -m brain_age_pred.scripts.train brain_age_pred/configs/brainagenext_snellius.yaml