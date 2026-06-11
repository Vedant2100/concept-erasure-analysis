#!/usr/bin/env bash
#SBATCH --job-name=fidelity
#SBATCH --output=slurm_fidelity_%j.log
#SBATCH --time=00:40:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

source ~/.bashrc
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate speed_env || conda activate mace_env

cd $SLURM_SUBMIT_DIR

# ================================================================
# Experiment 4 — Fidelity vs Identity.
# Pure re-analysis of already-generated images (experiment3/results/multi_concept).
# No diffusion generation, no checkpoints. Computes CLIP semantic drift AND LPIPS
# perceptual distance, paired same-seed against baseline, for retained canaries vs
# style-far controls. The finding is the CONTRAST: identity preserved (low CLIP)
# while fidelity degrades (LPIPS above controls).
# ================================================================

# LPIPS is the only extra dependency; setup_speed.sh installed it previously, but
# ensure it's present (no-op if already installed).
python -c "import lpips" 2>/dev/null || pip install -q lpips

echo "=== Running fidelity analysis ==="
python experiment3/scripts/analyze_fidelity.py \
    --root experiment3/results/multi_concept \
    --out_csv experiment3/results/fidelity/fidelity.csv

echo "=== Done. fidelity.csv written. Commit: git add experiment3/ && git commit ==="
