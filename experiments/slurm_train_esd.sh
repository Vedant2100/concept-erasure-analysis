#!/bin/bash
#SBATCH --job-name=train_esd
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Use absolute paths or $SLURM_SUBMIT_DIR for safety
cd $SLURM_SUBMIT_DIR

# 1. Setup Conda Environment
source ~/.bashrc
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Ensure we use an environment with diffusers
conda activate speed_env || conda activate mace_env

# 2. Clone ESD
if [ ! -d "../erasing" ]; then
    echo "Cloning ESD repository..."
    git clone https://github.com/rohitgandikota/erasing.git ../erasing
fi

cd ../erasing

# The updated ESD code requires diffusers, transformers, torch, accelerate.
pip install -r requirements.txt

# 3. Train ESD for Snoopy
echo "Training ESD on Snoopy (esd-x)..."
python esd_sd.py \
    --erase_concept 'Snoopy' \
    --train_method 'esd-x'

# The script outputs checkpoints by default in `compvis-models/Snoopy-esd-x/` or `models/Snoopy-esd-x`
# Since it was updated, it uses diffusers to save.
# 4. Train ESD for Van Gogh
echo "Training ESD on Van Gogh (esd-x)..."
python esd_sd.py \
    --erase_concept 'Van Gogh' \
    --train_method 'esd-x'

echo "ESD training complete."
