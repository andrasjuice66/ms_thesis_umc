#!/bin/bash
#SBATCH --job-name=sfcn
#SBATCH --partition=gpu_h100            
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END               

set -euo pipefail

# 1) activate your env
source ~/.bashrc
conda activate brain_age_310

# 2) (optional) copy large data to fast local SSD on the node
# SCRATCH=/scratch/$USER/$SLURM_JOB_ID
# mkdir -p $SCRATCH && cp -r /project/<proj>/data/dataset $SCRATCH

# 3) launch training exactly as on your laptop
cd $HOME/
python -m brain_age_pred.scripts.train brain_age_pred/configs/sfcn/sfcn_snellius.yaml