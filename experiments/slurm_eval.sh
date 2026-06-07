#!/usr/bin/env bash
#SBATCH --job-name=speed_eval
#SBATCH --output=slurm_eval_%j.log
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source ~/.bashrc
# Initialize conda properly for non-interactive shells
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

conda activate speed_env || conda activate mace_env

cd $SLURM_SUBMIT_DIR

echo "Starting Evaluation for Snoopy (Instance)..."
python experiments/eval_recovery.py --results_dir results/probe_ti/snoopy --target_concept "Snoopy" --anchor_concept "dog"

echo "Starting Evaluation for Van Gogh (Style)..."
python experiments/eval_recovery.py --results_dir results/probe_ti/vangogh --target_concept "Van Gogh" --anchor_concept "art"

echo "Evaluation complete! Check results/probe_ti/snoopy/evaluation_metrics.csv and results/probe_ti/vangogh/evaluation_metrics.csv"
