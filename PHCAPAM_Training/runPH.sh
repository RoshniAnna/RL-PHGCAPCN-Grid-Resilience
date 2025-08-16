#!/bin/bash
#SBATCH --job-name=PH_train
#SBATCH --output=PH.out
#SBATCH --error=PH.err
#SBATCH --partition=a30    
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-user=raj180002@utdallas.edu     
#SBATCH --mail-type=BEGIN,END,FAIL


module load miniconda
source ~/.bashrc
conda activate ONR_RL_Env

export CUDA_VISIBLE_DEVICES=0

# Run your training script
python train_PH_5.py
