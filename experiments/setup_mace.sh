#!/bin/bash
# setup_mace.sh

echo "Cloning MACE repository..."
if [ ! -d "../MACE" ]; then
    git clone https://github.com/Shilin-LU/MACE.git ../MACE
fi

echo "Installing MACE dependencies..."
cd ../MACE
# We assume the cluster environment handles most diffusers/torch requirements.
# If MACE has a requirements.txt, install it:
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "MACE setup complete."
