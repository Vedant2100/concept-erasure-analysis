#!/bin/bash
#SBATCH --job-name=train_mace
#SBATCH --output=slurm_train_mace_%j.log
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00

source ~/miniconda/etc/profile.d/conda.sh
conda activate base

echo "Starting MACE training for Snoopy..."
cd ../MACE
# Assuming a standard MACE training signature based on their repo structure
python train_mace.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --concept "Snoopy" \
  --output_dir "../MACE_weights/snoopy" \
  --max_train_steps 500

echo "Starting MACE training for Van Gogh..."
python train_mace.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --concept "Van Gogh style" \
  --output_dir "../MACE_weights/vangogh" \
  --max_train_steps 500

echo "MACE training completed."
