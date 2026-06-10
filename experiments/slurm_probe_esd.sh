#!/bin/bash
#SBATCH --job-name=probe_esd
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_probe_esd_%j.log

echo "Activating speed_env..."
source ~/miniconda/etc/profile.d/conda.sh
conda activate speed_env

# 1. Download the ESD Van Gogh weights from Baulab
mkdir -p checkpoints/esd
if [ ! -f "checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt" ]; then
    echo "Downloading ESD Van Gogh weights..."
    wget -q --show-progress "https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt" -O checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt
fi

echo "=== Compositional Probe: ESD (Van Gogh) ==="
python experiments/probe_compositional.py \
    --method esd \
    --ckpt_path checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt \
    --prompt_direct "a painting in the style of Van Gogh" \
    --prompt_synonym "a painting in the style of a post-impressionist Dutch painter" \
    --prompt_compositional "a painting with thick impasto brushstrokes and swirling night skies over a village" \
    --out_dir results/comp_vangogh

echo "=== TI Recovery: ESD (Van Gogh) ==="
python experiments/probe_textual_inversion.py \
    --method esd \
    --ckpt_path checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt \
    --reference_prompt "a painting in the style of Van Gogh" \
    --learned_token "<vangogh>" \
    --anchor_concept "art" \
    --template_type "style" \
    --budget_grid 0 50 200 500 \
    --out_dir results/ti_esd_vangogh

echo "Evaluating Metrics for ESD Van Gogh..."
python experiments/evaluate_metrics.py --probe ti --concept vangogh --method esd

echo "ESD Probes Completed!"
