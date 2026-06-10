#!/bin/bash
# setup_esd.sh

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
