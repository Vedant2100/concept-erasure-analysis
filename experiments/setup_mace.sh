#!/usr/bin/env bash
# setup_mace.sh — clone MACE and download checkpoints
set -e

echo "==> Cloning MACE..."
if [ ! -d "MACE" ]; then
    git clone https://github.com/Shilin-LU/MACE.git
fi

echo "==> Installing MACE dependencies..."
pip install -r MACE/requirements.txt

echo "==> Installing project dependencies..."
pip install -r requirements.txt

echo "==> Downloading MACE erased checkpoints from HuggingFace..."
python - <<'EOF'
from huggingface_hub import snapshot_download
import os

# MACE erased model for nudity (ESD-u equivalent but at scale)
print("Downloading MACE nudity-erased checkpoint...")
snapshot_download(
    repo_id="Shilin-LU/MACE_erased_nudity",
    local_dir="checkpoints/mace_nudity",
    ignore_patterns=["*.msgpack", "*.h5"],
)

# MACE artist-erased checkpoint (Van Gogh)
print("Downloading MACE Van Gogh-erased checkpoint...")
snapshot_download(
    repo_id="Shilin-LU/MACE_erased_vangogh",
    local_dir="checkpoints/mace_vangogh",
    ignore_patterns=["*.msgpack", "*.h5"],
)

print("All checkpoints downloaded.")
EOF

echo ""
echo "==> Setup complete. Run experiments with:"
echo "    python experiments/exp1_compositional.py"
echo "    python experiments/exp2_collateral.py"
