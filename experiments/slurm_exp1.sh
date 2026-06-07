#!/usr/bin/env bash
#SBATCH --job-name=mace_exp1
#SBATCH --output=slurm_exp1_%j.log
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# --- ENVIRONMENT SETUP ---
# Activate the mace_env Conda environment
source ~/.bashrc
conda activate mace_env

echo "Starting Experiment 1: Compositional Evasion..."
# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

python experiments/exp1_compositional.py
echo "Experiment 1 Complete."
