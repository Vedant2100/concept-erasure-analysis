#!/usr/bin/env bash
#SBATCH --job-name=dpa_ablation
#SBATCH --output=slurm_dpa_ablation_%j.log
#SBATCH --time=01:30:00
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
# Experiment 3.5 — DPA / Prior-Knowledge-Refinement ablation at N=40.
#
# At the load where Pissarro (an explicit retain-set member) leaks under the full
# method, does SPEED's refinement machinery HELP or HURT that entangled member?
#   speed_40c          full (IPF + DPA)        -> already built by rank-sat run
#   speed_40c_nodpa    IPF on, DPA off         -> --aug_num 10 --disable_dpa
#   speed_40c_norefine neither IPF nor DPA     -> --aug_num 0
# Only the two new checkpoints are built here; baseline + speed_40c images already
# exist under experiment3/results/rank_saturation/.
# ================================================================

# Same nested impressionist cluster as rank-sat; we only need N=40.
N5="Claude Monet, Pierre-Auguste Renoir, Edgar Degas, Edouard Manet, Alfred Sisley"
N10="$N5, Berthe Morisot, Gustave Caillebotte, Mary Cassatt, Paul Cezanne, Vincent van Gogh"
N20="$N10, Paul Signac, Henri de Toulouse-Lautrec, Pierre Bonnard, Edouard Vuillard, Henri-Edmond Cross, Armand Guillaumin, Childe Hassam, John Singer Sargent, Eugene Boudin, Johan Jongkind"
N40="$N20, Maximilien Luce, Theo van Rysselberghe, Albert Dubois-Pillet, Charles Angrand, Henri Martin, Gustave Loiseau, Maxime Maufra, Albert Lebourg, Henri Le Sidaner, Frederic Bazille, Jean Beraud, Federico Zandomeneghi, Giovanni Boldini, Joaquin Sorolla, Max Liebermann, Lovis Corinth, Konstantin Korovin, Valentin Serov, Frits Thaulow, Peder Severin Kroyer"

build () {  # build <save_subdir> <extra_flags>
    local outdir="checkpoints/speed/rank/$1"; shift
    mkdir -p "$outdir"
    cd SPEED_repo
    python train_erase_null.py \
        --baseline SPEED \
        --target_concepts "$N40" --anchor_concepts "art" \
        --retain_path "data/style.csv" --heads "concept" \
        --save_path "../$outdir" --file_name "weight" \
        --params V --aug_num 10 --threshold 1e-1 "$@"
    cd ..
}

echo "=== Building DPA-off (IPF kept) checkpoint ==="
build "40c_nodpa" --disable_dpa
echo "=== Building refinement-off (aug_num 0) checkpoint ==="
# aug_num 0 turns off both IPF filtering and DPA; pass it explicitly.
mkdir -p checkpoints/speed/rank/40c_norefine
cd SPEED_repo
python train_erase_null.py \
    --baseline SPEED \
    --target_concepts "$N40" --anchor_concepts "art" \
    --retain_path "data/style.csv" --heads "concept" \
    --save_path "../checkpoints/speed/rank/40c_norefine" --file_name "weight" \
    --params V --aug_num 0 --threshold 1e-1
cd ..

echo "=== Generating (baseline + speed_40c already exist and will be skipped) ==="
python experiment3/scripts/probe_rank_saturation.py --method baseline          --out_dir experiment3/results/rank_saturation
python experiment3/scripts/probe_rank_saturation.py --method speed_40c_nodpa    --ckpt checkpoints/speed/rank/40c_nodpa/weight.pt    --out_dir experiment3/results/rank_saturation
python experiment3/scripts/probe_rank_saturation.py --method speed_40c_norefine --ckpt checkpoints/speed/rank/40c_norefine/weight.pt --out_dir experiment3/results/rank_saturation

echo "=== Analyzing DPA ablation ==="
python experiment3/scripts/analyze_dpa_ablation.py \
    --root experiment3/results/rank_saturation \
    --out_csv experiment3/results/rank_saturation/dpa_ablation.csv

echo "=== Done. dpa_ablation.csv written. Commit: git add experiment3/ SPEED_repo/train_erase_null.py ==="
