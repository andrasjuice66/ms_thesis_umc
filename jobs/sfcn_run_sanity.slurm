#!/bin/bash
#SBATCH --job-name=sfcn
#SBATCH --partition=gpu_a100            
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END               

# set -euo pipefail

# 1) activate your env
source ~/.bashrc
conda activate thesis

# 2) (optional) copy large data to fast local SSD on the node
# SCRATCH=/scratch/$USER/$SLURM_JOB_ID
# mkdir -p $SCRATCH && cp -r /project/<proj>/data/dataset $SCRATCH

# 3) launch training exactly as on your laptop
cd $HOME/
# python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_overfit.yaml

# python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_t1.yaml

# python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_30_40.yaml

# python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_60_70.yaml

# python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_t2.yaml

python -m brain_age_pred.scripts.train brain_age_pred/configs/sanity/sfcn_snellius_t1_30_40.yaml
