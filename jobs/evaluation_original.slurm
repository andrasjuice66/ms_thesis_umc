#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --partition=gpu_h100            
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END    
#SBATCH --mail-user=a.g.joos@student.vu.nl

set -x
set -o pipefail

source ~/.bashrc
conda activate thesis

# SCRATCH=/scratch/$USER/$SLURM_JOB_ID
# mkdir -p $SCRATCH && cp -r /project/<proj>/data/dataset $SCRATCH

cd $HOME/
python -m brain_age_pred.scripts.inference_sfcn 

python -m brain_age_pred.scripts.inference_brainagenext --config brain_age_pred/configs/evaluate/brainagenext_original.yaml

