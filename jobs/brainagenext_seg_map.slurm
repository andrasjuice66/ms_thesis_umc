#!/bin/bash
#SBATCH --job-name=brainagenext
#SBATCH --partition=gpu_a100            
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END    
#SBATCH --mail-user=a.g.joos@vu.student.nl

set -x
set -o pipefail

# 1) activate your env
source ~/.bashrc
conda activate brain_age_310

# 2) (optional) copy large data to fast local SSD on the node
# SCRATCH=/scratch/$USER/$SLURM_JOB_ID
# mkdir -p $SCRATCH && cp -r /project/<proj>/data/dataset $SCRATCH

cd $HOME/

python -m brain_age_pred.scripts.train_seg_map brain_age_pred/configs/brainagenext_seg_map/brainagenext_seg_map_augmented.yaml




