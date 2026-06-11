#!/usr/bin/env bash
#SBATCH --job-name=probe_neighbor
#SBATCH --output=slurm_probe_neighbor_%j.log
#SBATCH --time=03:00:00
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

# Sanity check: verify required checkpoints exist before spending GPU time
if [ ! -f "checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt" ]; then
    echo "ERROR: ESD-x Van Gogh checkpoint missing. Run setup_esd_neighbor.sh first."
    exit 1
fi
if [ ! -f "checkpoints/speed/few-concept/style/Van Gogh.pt" ]; then
    echo "ERROR: SPEED Van Gogh checkpoint missing. Run setup_speed.sh first."
    exit 1
fi
if [ ! -f "checkpoints/speed/few-concept/instance/Snoopy.pt" ]; then
    echo "ERROR: SPEED Snoopy checkpoint missing. Run setup_speed.sh first."
    exit 1
fi

# ================================================================
# Van Gogh — three-way comparison (baseline, ESD-x, SPEED)
# ================================================================

echo "=== Van Gogh: Baseline ==="
python experiments/probe_neighbor_damage.py \
    --method baseline \
    --concept vangogh \
    --out_dir results/neighbor_damage

echo "=== Van Gogh: ESD-x ==="
python experiments/probe_neighbor_damage.py \
    --method esd_x \
    --ckpt "checkpoints/esd/diffusers-VanGogh-ESDx1-UNET.pt" \
    --concept vangogh \
    --out_dir results/neighbor_damage

echo "=== Van Gogh: SPEED ==="
python experiments/probe_neighbor_damage.py \
    --method speed \
    --ckpt "checkpoints/speed/few-concept/style/Van Gogh.pt" \
    --concept vangogh \
    --out_dir results/neighbor_damage

# ================================================================
# Snoopy — two-way comparison (baseline, SPEED only)
# No ESD-x Snoopy checkpoint exists from Baulab — they only released
# five artist style checkpoints + nudity. Snoopy is instance erasure
# which is outside ESD's original released scope.
# ================================================================

echo "=== Snoopy: Baseline ==="
python experiments/probe_neighbor_damage.py \
    --method baseline \
    --concept snoopy \
    --out_dir results/neighbor_damage

echo "=== Snoopy: SPEED ==="
python experiments/probe_neighbor_damage.py \
    --method speed \
    --ckpt "checkpoints/speed/few-concept/instance/Snoopy.pt" \
    --concept snoopy \
    --out_dir results/neighbor_damage

echo "All neighbor damage probes complete."
