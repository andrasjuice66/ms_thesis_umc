#!/bin/bash
#SBATCH --job-name=sfcn_synth
#SBATCH --partition=gpu_a100            
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END    
#SBATCH --mail-user=a.g.joos@student.vu.nl

set -x
set -o pipefail

# 1) activate your env
source ~/.bashrc
conda activate thesis

# 2) (optional) copy large data to fast local SSD on the node
# SCRATCH=/scratch/$USER/$SLURM_JOB_ID
# mkdir -p $SCRATCH && cp -r /project/<proj>/data/dataset $SCRATCH

# 3) launch training exactly as on your laptop
cd $HOME/

python -m brain_age_pred.scripts.train_synth brain_age_pred/configs/segmented/sfcn_class_segmented_snell.yaml









