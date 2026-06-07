#!/usr/bin/env bash
# setup_speed.sh — install SPEED dependencies and download released checkpoints
set -e

echo "==> Ensuring SPEED repo is cloned..."
if [ ! -d "SPEED_repo" ]; then
    git clone https://github.com/Ouxiang-Li/SPEED.git SPEED_repo
fi

echo "==> Installing dependencies..."
# Assuming mace_env or speed_env is already activated via SLURM
pip install torch==2.3.0 torchvision==0.18.0
pip install -r SPEED_repo/requirements.txt
# Additional deps for our probes
pip install lpips

echo "==> Downloading SPEED erased checkpoints from HuggingFace..."
mkdir -p checkpoints/speed

python - <<'EOF'
from huggingface_hub import hf_hub_download
import os

print("Downloading SPEED Snoopy (instance) checkpoint...")
hf_hub_download(
    repo_id="lioooox/SPEED",
    filename="few-concept/instance/Snoopy.pt",
    local_dir="checkpoints/speed",
    local_dir_use_symlinks=False
)

print("Downloading SPEED Van Gogh (style) checkpoint...")
hf_hub_download(
    repo_id="lioooox/SPEED",
    filename="few-concept/style/Van Gogh.pt",
    local_dir="checkpoints/speed",
    local_dir_use_symlinks=False
)

print("All checkpoints downloaded.")
EOF

echo "==> Setup complete."
