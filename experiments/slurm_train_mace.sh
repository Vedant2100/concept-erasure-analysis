#!/usr/bin/env bash
#SBATCH --job-name=train_mace
#SBATCH --output=slurm_train_mace_%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/.bashrc
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate speed_env || conda activate mace_env

cd $SLURM_SUBMIT_DIR

echo "Starting MACE training for Snoopy..."
cd MACE

# Ensure dependencies are installed in the activated conda environment
pip install -r requirements.txt
pip install omegaconf openai

# MACE's actual training script is training.py
python training.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --concept "Snoopy" \
  --output_dir "../MACE_weights/snoopy" \
  --max_train_steps 500

echo "Starting MACE training for Van Gogh..."
python training.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --concept "Van Gogh style" \
  --output_dir "../MACE_weights/vangogh" \
  --max_train_steps 500

echo "MACE training completed."
