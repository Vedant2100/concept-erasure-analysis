#!/usr/bin/env bash
#SBATCH --job-name=rank_saturation
#SBATCH --output=slurm_rank_saturation_%j.log
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

# ================================================================
# Experiment 3.4 — Null-space rank saturation under concentrated mass erasure.
#
# Erase a NESTED, same-movement cluster of impressionist/post-impressionist
# painters (N = 5,10,20,40). Each list is a strict superset of the previous so the
# 5->10->20->40 trend is clean. Gauguin / Seurat / Pissarro (held-out canaries),
# Rembrandt / Hokusai (style-far controls) are NEVER in any list.
#
# SPEED is training-free, so each checkpoint is ~seconds even at N=40. Settings are
# identical across N (params=V, aug_num=10, threshold=1e-1) — the ONLY variable is
# how many concentrated concepts are erased.
# ================================================================

# --- Nested impressionist-family erase lists (none are canaries/controls) ---
N5="Claude Monet, Pierre-Auguste Renoir, Edgar Degas, Edouard Manet, Alfred Sisley"
N10="$N5, Berthe Morisot, Gustave Caillebotte, Mary Cassatt, Paul Cezanne, Vincent van Gogh"
N20="$N10, Paul Signac, Henri de Toulouse-Lautrec, Pierre Bonnard, Edouard Vuillard, Henri-Edmond Cross, Armand Guillaumin, Childe Hassam, John Singer Sargent, Eugene Boudin, Johan Jongkind"
N40="$N20, Maximilien Luce, Theo van Rysselberghe, Albert Dubois-Pillet, Charles Angrand, Henri Martin, Gustave Loiseau, Maxime Maufra, Albert Lebourg, Henri Le Sidaner, Frederic Bazille, Jean Beraud, Federico Zandomeneghi, Giovanni Boldini, Joaquin Sorolla, Max Liebermann, Lovis Corinth, Konstantin Korovin, Valentin Serov, Frits Thaulow, Peder Severin Kroyer"

build_ckpt () {
    local targets="$1"; local outdir="$2"
    mkdir -p "$outdir"
    cd SPEED_repo
    python train_erase_null.py \
        --baseline SPEED \
        --target_concepts "$targets" --anchor_concepts "art" \
        --retain_path "data/style.csv" --heads "concept" \
        --save_path "../$outdir" --file_name "weight" \
        --params V --aug_num 10 --threshold 1e-1
    cd ..
}

echo "=== Building checkpoints (5,10,20,40 concentrated impressionists) ==="
build_ckpt "$N5"  "checkpoints/speed/rank/5c"
build_ckpt "$N10" "checkpoints/speed/rank/10c"
build_ckpt "$N20" "checkpoints/speed/rank/20c"
build_ckpt "$N40" "checkpoints/speed/rank/40c"

echo "=== Generating held-out test images for each model ==="
python experiment3/scripts/probe_rank_saturation.py --method baseline  --out_dir experiment3/results/rank_saturation
python experiment3/scripts/probe_rank_saturation.py --method speed_5c  --ckpt checkpoints/speed/rank/5c/weight.pt  --out_dir experiment3/results/rank_saturation
python experiment3/scripts/probe_rank_saturation.py --method speed_10c --ckpt checkpoints/speed/rank/10c/weight.pt --out_dir experiment3/results/rank_saturation
python experiment3/scripts/probe_rank_saturation.py --method speed_20c --ckpt checkpoints/speed/rank/20c/weight.pt --out_dir experiment3/results/rank_saturation
python experiment3/scripts/probe_rank_saturation.py --method speed_40c --ckpt checkpoints/speed/rank/40c/weight.pt --out_dir experiment3/results/rank_saturation

echo "=== Analyzing CLIP drift across the N sweep ==="
python experiment3/scripts/analyze_rank_saturation.py \
    --root experiment3/results/rank_saturation \
    --out_csv experiment3/results/rank_saturation/rank_drift.csv

echo "=== Done. Results + rank_drift.csv in experiment3/results/rank_saturation/ ==="
echo "Commit: git add experiment3/ && git commit -m 'Exp 3.4 rank-saturation results'"
