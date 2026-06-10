#!/bin/bash
# setup_esd.sh — Pre-cache ESD erased model weights from HuggingFace

source ~/.bashrc
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate speed_env || conda activate mace_env

echo "Pre-caching ESD models from HuggingFace..."
python -c "
from diffusers import StableDiffusionPipeline
import torch

print('Downloading erasing-snoopy...')
StableDiffusionPipeline.from_pretrained('rohitgandikota/erasing-snoopy', torch_dtype=torch.float16)

print('Downloading erasing-vangogh...')
StableDiffusionPipeline.from_pretrained('rohitgandikota/erasing-vangogh', torch_dtype=torch.float16)

print('ESD models successfully cached.')
"
