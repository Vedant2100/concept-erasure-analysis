#!/usr/bin/env bash
#SBATCH --job-name=speed_ti
#SBATCH --output=slurm_probe_ti_%j.log
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/.bashrc
# Initialize conda properly for non-interactive shells
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

conda activate speed_env || conda activate mace_env

cd $SLURM_SUBMIT_DIR

echo "Starting Textual Inversion Recovery Probe for Snoopy (Instance)..."
python experiments/probe_textual_inversion.py \
    --speed_ckpt checkpoints/speed/Snoopy.pt \
    --reference_prompt "a photo of Snoopy" \
    --learned_token "<snoopy>" \
    --anchor_concept "dog" \
    --template_type "instance" \
    --budget_grid 0 50 200 500 1000 \
    --out_dir results/probe_ti/snoopy

echo "Starting Textual Inversion Recovery Probe for Van Gogh (Style)..."
python experiments/probe_textual_inversion.py \
    --speed_ckpt "checkpoints/speed/Van Gogh.pt" \
    --reference_prompt "a painting in the style of Van Gogh" \
    --learned_token "<vangogh>" \
    --anchor_concept "art" \
    --template_type "style" \
    --budget_grid 0 50 200 500 1000 \
    --out_dir results/probe_ti/vangogh

echo "Probes complete!"
