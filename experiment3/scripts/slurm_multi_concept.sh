#!/usr/bin/env bash
#SBATCH --job-name=multi_concept_collapse
#SBATCH --output=slurm_multi_concept_%j.log
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

source ~/.bashrc
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate speed_env || conda activate mace_env

cd $SLURM_SUBMIT_DIR

# Verify required checkpoints exist
if [ ! -f "checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt" ]; then
    echo "ERROR: ESD-x checkpoint missing. Run setup_esd_neighbor.sh first."
    exit 1
fi

mkdir -p checkpoints/speed/multi/1concept
mkdir -p checkpoints/speed/multi/2concept
mkdir -p checkpoints/speed/multi/3concept

# ================================================================
# Step 1: Generate ALL THREE SPEED checkpoints with IDENTICAL settings.
# SPEED is training-free — each takes ~60 seconds.
#
# CRITICAL: we build the 1-concept checkpoint here rather than reusing the
# released few-concept/style/Van Gogh.pt. All three (1c/2c/3c) must come from the
# same script, retain set (style.csv), and hyperparameters (V, aug_num=10,
# threshold=1e-1) so the ONLY variable across columns is the number of erased
# concepts. Reusing the released checkpoint would confound concept-count with
# checkpoint provenance and invalidate the 1c->2c->3c trend.
# ================================================================

echo "=== Building 1-concept checkpoint: Van Gogh ==="
cd SPEED_repo
python train_erase_null.py \
    --baseline SPEED \
    --target_concepts "Van Gogh" \
    --anchor_concepts "art" \
    --retain_path "data/style.csv" --heads "concept" \
    --save_path "../checkpoints/speed/multi/1concept" \
    --file_name "weight" \
    --params V --aug_num 10 --threshold 1e-1
cd ..

echo "=== Building 2-concept checkpoint: Van Gogh + Picasso ==="
cd SPEED_repo
python train_erase_null.py \
    --baseline SPEED \
    --target_concepts "Van Gogh, Picasso" \
    --anchor_concepts "art" \
    --retain_path "data/style.csv" --heads "concept" \
    --save_path "../checkpoints/speed/multi/2concept" \
    --file_name "weight" \
    --params V --aug_num 10 --threshold 1e-1
cd ..

echo "=== Building 3-concept checkpoint: Van Gogh + Picasso + Monet ==="
cd SPEED_repo
python train_erase_null.py \
    --baseline SPEED \
    --target_concepts "Van Gogh, Picasso, Monet" \
    --anchor_concepts "art" \
    --retain_path "data/style.csv" --heads "concept" \
    --save_path "../checkpoints/speed/multi/3concept" \
    --file_name "weight" \
    --params V --aug_num 10 --threshold 1e-1
cd ..

# ================================================================
# Step 2: Generate images for all test artists, all 4 methods
# ================================================================

echo "=== Generating: Baseline ==="
python experiment3/scripts/probe_multi_concept.py \
    --method baseline \
    --out_dir experiment3/results/multi_concept

echo "=== Generating: SPEED 1-concept (Van Gogh only) ==="
python experiment3/scripts/probe_multi_concept.py \
    --method speed_1c \
    --ckpt "checkpoints/speed/multi/1concept/weight.pt" \
    --out_dir experiment3/results/multi_concept

echo "=== Generating: SPEED 2-concept (Van Gogh + Picasso) ==="
python experiment3/scripts/probe_multi_concept.py \
    --method speed_2c \
    --ckpt "checkpoints/speed/multi/2concept/weight.pt" \
    --out_dir experiment3/results/multi_concept

echo "=== Generating: SPEED 3-concept (Van Gogh + Picasso + Monet) ==="
python experiment3/scripts/probe_multi_concept.py \
    --method speed_3c \
    --ckpt "checkpoints/speed/multi/3concept/weight.pt" \
    --out_dir experiment3/results/multi_concept

echo "=== Generating: ESD-x (Van Gogh, single concept, no retain set) ==="
python experiment3/scripts/probe_multi_concept.py \
    --method esd_x \
    --ckpt "checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt" \
    --out_dir experiment3/results/multi_concept

# ================================================================
# Step 3: Measure CLIP image-image drift across the 1c->2c->3c progression.
# This is the metric that decides the experiment — NOT pixel MSE.
# ================================================================
echo "=== Analyzing CLIP drift (canaries vs style-far controls) ==="
python experiment3/scripts/analyze_clip_drift.py \
    --root experiment3/results/multi_concept \
    --out_csv experiment3/results/multi_concept/clip_drift.csv

echo "=== All multi-concept probes complete. ==="
echo "Results in experiment3/results/multi_concept/  (images + clip_drift.csv)"
echo "Commit with: git add experiment3/results/multi_concept/ experiment3/scripts/ && git commit -m 'Add multi-concept null-space collapse experiment'"
