#!/bin/bash
#SBATCH --job-name=dl_tabular_resnet
#SBATCH --partition=gpu_a100            
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END    
#SBATCH --mail-user=a.g.joos@vu.student.nl

set -x
set -o pipefail

# Activate environment
source ~/.bashrc
conda activate brain_age_310

# Change to home directory
cd $HOME/

# Run deep learning tabular training
python -m brain_age_pred.scripts.train_dl_tabular brain_age_pred/configs/deep_learning_tabular/dl_tabular.yaml

