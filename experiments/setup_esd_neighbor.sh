#!/usr/bin/env bash
# setup_esd_neighbor.sh
# Downloads the official ESD-x Van Gogh UNet checkpoint from Baulab.
#
# Source: https://erasing.baulab.info/weights/esd_models/art/
# File:   diffusers-VanGogh-ESDx1-UNET.pt
# Size:   ~3.2 GB  — verify disk space before running.
#
# Note: the earlier setup_esd.sh references rohitgandikota/erasing-snoopy and
# rohitgandikota/erasing-vangogh on HuggingFace, which do not exist. This
# script replaces it with the correct download from the official project page.
set -e

mkdir -p checkpoints/esd

ESD_URL="https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt"
ESD_PATH="checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt"

if [ -f "$ESD_PATH" ]; then
    echo "ESD-x Van Gogh checkpoint already present at $ESD_PATH — skipping download."
else
    echo "Downloading ESD-x Van Gogh checkpoint (~3.2 GB)..."
    # -L follows redirects; -C - resumes an interrupted download
    curl -L -C - "$ESD_URL" -o "$ESD_PATH"
    echo "Saved to $ESD_PATH"
fi

echo "Setup complete."
