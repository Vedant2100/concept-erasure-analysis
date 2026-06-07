#!/usr/bin/env bash
#SBATCH --job-name=mace_setup
#SBATCH --output=slurm_setup_%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# --- ENVIRONMENT SETUP ---
# Setup Conda environment
source ~/.bashrc
# Create a new environment called 'mace_env' with Python 3.10 if it doesn't exist
conda create -n mace_env python=3.10 -y
conda activate mace_env

echo "Starting MACE Setup..."
# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

bash experiments/setup_mace.sh
echo "Setup Complete."
