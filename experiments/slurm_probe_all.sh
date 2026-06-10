#!/bin/bash
#SBATCH --job-name=probe_all
#SBATCH --output=slurm_probe_all_%j.log
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00

source ~/miniconda/etc/profile.d/conda.sh
conda activate base

cd ~/concept-erasure-project/experiments

echo "Running Baseline Probes..."
python probe_compositional.py --method baseline \
    --prompt_direct "a photo of Snoopy" \
    --prompt_synonym "a photo of the beagle from Peanuts" \
    --prompt_compositional "a white dog with black ears sleeping on a red doghouse" \
    --out_dir "../results/comp_snoopy"

python probe_textual_inversion.py --method baseline \
    --reference_prompt "a photo of Snoopy" \
    --learned_token "<snoopy>" \
    --anchor_concept "dog" \
    --template_type "instance" \
    --out_dir "../results/ti_baseline_snoopy"

echo "Running SPEED Probes..."
python probe_compositional.py --method speed --ckpt_path "../checkpoints/snoopy.pt" \
    --prompt_direct "a photo of Snoopy" \
    --prompt_synonym "a photo of the beagle from Peanuts" \
    --prompt_compositional "a white dog with black ears sleeping on a red doghouse" \
    --out_dir "../results/comp_snoopy"

python probe_textual_inversion.py --method speed --ckpt_path "../checkpoints/snoopy.pt" \
    --reference_prompt "a photo of Snoopy" \
    --learned_token "<snoopy>" \
    --anchor_concept "dog" \
    --template_type "instance" \
    --out_dir "../results/ti_speed_snoopy"

echo "Running ESD Probes..."
python probe_compositional.py --method esd --ckpt_path "rohitgandikota/erasing-snoopy" \
    --prompt_direct "a photo of Snoopy" \
    --prompt_synonym "a photo of the beagle from Peanuts" \
    --prompt_compositional "a white dog with black ears sleeping on a red doghouse" \
    --out_dir "../results/comp_snoopy"

python probe_textual_inversion.py --method esd --ckpt_path "rohitgandikota/erasing-snoopy" \
    --reference_prompt "a photo of Snoopy" \
    --learned_token "<snoopy>" \
    --anchor_concept "dog" \
    --template_type "instance" \
    --out_dir "../results/ti_esd_snoopy"

echo "Running MACE Probes..."
python probe_compositional.py --method mace --ckpt_path "../MACE_weights/snoopy" \
    --prompt_direct "a photo of Snoopy" \
    --prompt_synonym "a photo of the beagle from Peanuts" \
    --prompt_compositional "a white dog with black ears sleeping on a red doghouse" \
    --out_dir "../results/comp_snoopy"

python probe_textual_inversion.py --method mace --ckpt_path "../MACE_weights/snoopy" \
    --reference_prompt "a photo of Snoopy" \
    --learned_token "<snoopy>" \
    --anchor_concept "dog" \
    --template_type "instance" \
    --out_dir "../results/ti_mace_snoopy"

echo "All Probes Completed!"
