#!/usr/bin/env bash
#SBATCH --job-name=mace_exp2
#SBATCH --output=slurm_exp2_%j.log
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

echo "Starting Experiment 2: Semantic Collateral Damage..."
# Ensure we are in the project root directory
cd $SLURM_SUBMIT_DIR

# Generate the erased church checkpoint first if it doesn't exist
if [ ! -d "checkpoints/mace_church" ]; then
    echo "Church checkpoint not found. Generating it now..."
    python MACE/generate_erased.py --concept 'church' --output checkpoints/mace_church
fi

python experiments/exp2_collateral.py
echo "Experiment 2 Complete."
