#!/bin/bash
#SBATCH --job-name=evaluation_all
#SBATCH --partition=gpu_h100            
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/%u/run_logs/%x_%j.out   
#SBATCH --error=/home/%u/run_logs/%x_%j.err
#SBATCH --mail-type=FAIL,END    
#SBATCH --mail-user=a.g.joos@student.vu.nl

set -x
set -o pipefail

source ~/.bashrc
conda activate brain_age_310

cd $HOME/

# Common config file for all evaluations
CONFIG_FILE="brain_age_pred/configs/evaluate/evaluate.yaml"

#echo "=== Starting BrainAgeNeXt Model Evaluations"

# BrainAgeNeXt Baseline
#echo "--- Evaluating BrainAgeNeXt Baseline ---"
#python -m brain_age_pred.scripts.evaluate \
#    --model BrainAgeNeXt \
#    --model_path /home/ajoos/brain_age_pred/output/checkpoints/brainagenext_baseline/brainagenext_baseline_best_mae.pt \
#    --config $CONFIG_FILE

# BrainAgeNeXt Domain Randomization
#echo "--- Evaluating BrainAgeNeXt Domain Randomization ---"
#python -m brain_age_pred.scripts.evaluate \
#    --model BrainAgeNeXt \
#    --model_path /home/ajoos/brain_age_pred/output/checkpoints/brainagenext_dom_rand/brainagenext_dom_rand_best_mae.pt \
#    --config $CONFIG_FILE

# BrainAgeNeXt Domain Randomization + Tumor
#echo "--- Evaluating BrainAgeNeXt Domain Randomization + Tumor ---"
#python -m brain_age_pred.scripts.evaluate \
#    --model BrainAgeNeXt \
#    --model_path /home/ajoos/brain_age_pred/output/checkpoints/brainagenext_dom_rand_tumor/brainagenext_dom_rand_tumor_best_mae.pt \
#    --config $CONFIG_FILE

echo "=== Starting SFCN_Class Model Evaluations"

# SFCN_Class Baseline
echo "--- Evaluating SFCN_Class Baseline ---"
python -m brain_age_pred.scripts.evaluate \
    --model SFCN_Class \
    --model_path /home/ajoos/brain_age_pred/output/checkpoints/sfcn_class_baseline/sfcn_class_baseline_epoch144.pt \
    --config $CONFIG_FILE

# SFCN_Class Domain Randomization
echo "--- Evaluating SFCN_Class Domain Randomization ---"
python -m brain_age_pred.scripts.evaluate \
    --model SFCN_Class \
    --model_path /home/ajoos/brain_age_pred/output/checkpoints/sfcn_class_dom_rand/sfcn_class_dom_rand_best_mae.pt \
    --config $CONFIG_FILE

# SFCN_Class Domain Randomization + Tumor
echo "--- Evaluating SFCN_Class Domain Randomization + Tumor ---"
python -m brain_age_pred.scripts.evaluate \
    --model SFCN_Class \
    --model_path /home/ajoos/brain_age_pred/output/checkpoints/sfcn_class_dom_rand_tumor/sfcn_class_dom_rand_tumor_best_mae.pt \
    --config $CONFIG_FILE

echo "=== All Model Evaluations Complete" 