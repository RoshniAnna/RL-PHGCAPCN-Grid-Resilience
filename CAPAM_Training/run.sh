#!/bin/bash
#SBATCH --job-name=capam_train
#SBATCH --output=capam_train.out
#SBATCH --error=capam_train.err
#SBATCH --partition=a30          # or use h100 if needed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

module load miniconda
source ~/.bashrc
conda activate ONR_RL_Env

# Optional: for debugging
nvidia-smi
export CUDA_VISIBLE_DEVICES=0

# Run your script
python train_capam.py
