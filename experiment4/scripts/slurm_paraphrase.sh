#!/usr/bin/env bash
#SBATCH --job-name=paraphrase
#SBATCH --output=slurm_paraphrase_%j.log
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

# ================================================================
# Experiment 4 — Style-Paraphrase Evasion.
#
# SPEED's robustness evaluation (Appendix D.4) only tests evasion
# against nudity concepts (MMA, Ring-A-Bell, UnlearnDiff). Style
# robustness under paraphrasing is completely unevaluated — the
# exact same "test the regime they skipped" move as rank saturation.
#
# We test: can you evoke Van Gogh's visual style without using his
# name? If yes → SPEED's style erasure is trivially bypassed.
#
# Reuses existing checkpoints:
#   - SPEED 1c Van Gogh: checkpoints/speed/multi/1concept/weight.pt
#   - ESD-x Van Gogh:    checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt
# No new checkpoint generation needed.
# ================================================================

# --- Verify checkpoints exist ---
SPEED_CKPT="checkpoints/speed/multi/1concept/weight.pt"
ESD_CKPT="checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt"

if [ ! -f "$SPEED_CKPT" ]; then
    echo "WARNING: SPEED checkpoint not found at $SPEED_CKPT"
    echo "Building 1-concept Van Gogh checkpoint..."
    mkdir -p checkpoints/speed/multi/1concept
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
fi

if [ ! -f "$ESD_CKPT" ]; then
    echo "ERROR: ESD-x checkpoint missing at $ESD_CKPT"
    echo "Run: python experiment3/scripts/setup_esd_neighbor.sh first."
    exit 1
fi

echo "=== Checkpoints verified ==="
echo "  SPEED: $SPEED_CKPT"
echo "  ESD-x: $ESD_CKPT"

# ================================================================
# Step 1: Generate images — baseline, SPEED, ESD-x
# 13 prompts × 4 seeds × 3 methods = 156 images (~25 min)
# ================================================================

echo ""
echo "=== Generating: Baseline ==="
python experiment4/scripts/probe_paraphrase.py \
    --method baseline \
    --out_dir experiment4/results/paraphrase

echo ""
echo "=== Generating: SPEED 1c (Van Gogh erased) ==="
python experiment4/scripts/probe_paraphrase.py \
    --method speed_1c \
    --ckpt "$SPEED_CKPT" \
    --out_dir experiment4/results/paraphrase

echo ""
echo "=== Generating: ESD-x (Van Gogh erased) ==="
python experiment4/scripts/probe_paraphrase.py \
    --method esd_x \
    --ckpt "$ESD_CKPT" \
    --out_dir experiment4/results/paraphrase

# ================================================================
# Step 2: Analyze — CLIP Van-Gogh-ness + image drift
# ================================================================

echo ""
echo "=== Analyzing: CLIP Van-Gogh-ness & drift ==="
python experiment4/scripts/analyze_paraphrase.py \
    --root experiment4/results/paraphrase \
    --out_csv experiment4/results/paraphrase/evasion.csv

echo ""
echo "=== Done. evasion.csv written. ==="
echo "Commit: git add experiment4/ && git commit -m 'Exp 4: paraphrase evasion results'"
