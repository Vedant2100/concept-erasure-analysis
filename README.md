# Erasing Concepts from Diffusion Models — Semantic Neighbor Damage

**Course:** EE243  
**Student:** Vedant  
**Frontier paper:** SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models (Li et al., ICLR 2026)

🌐 **[Project Webpage](https://vedant2100.github.io/concept-erasure-analysis)**

---

## Experiment 3: Semantic Neighbor Collateral Damage (The "Blast Radius" Test)

This branch hosts the codebase and visual results specifically for **Experiment 3** (Semantic Neighbor Collateral Damage).

We investigate the **retain-set horizon** of SPEED's null-space projection:
* **The Goal:** Test if SPEED's null-space projection edits bleed into semantic neighbors (similar artists) that fall outside the explicit protected list (retain set).
* **Key Discovery:** It is not a binary damage-vs-no-damage landscape, but a spectrum. Artists with high mathematical footprints (displacement) under SPEED's edit suffer noticeable visual style suppression (e.g., Theo van Rysselberghe and Jan Toorop) even when not in the retain set, while low-similarity adjacent artists (Monticelli, van Rappard) remain perfectly preserved.

---

## Branch Structure

* `experiments/`
  * `footprint_analysis.py` — Calculates mathematical footprint / weight displacement for unprotected style concepts.
  * `probe_neighbor_damage.py` — Evaluates neighbor visual damage by generating images from baseline, SPEED, and ESD-x.
  * `neighbor_prompts.json` — Prompt library containing target artists.
  * `setup_speed.sh` / `setup_esd_neighbor.sh` — Downloads weights.
  * `slurm_probe_neighbor.sh` — Executes the neighbor damage probe on cluster.
* `results/neighbor_damage/` — PNG outputs for baseline, SPEED, and ESD-x across target artists.

To run the footprint analysis:
```bash
python experiments/footprint_analysis.py \
    --speed_ckpt checkpoints/speed/few-concept/style/Van\ Gogh.pt \
    --base_model CompVis/stable-diffusion-v1-4 \
    --retain_set SPEED_repo/data/style.csv \
    --out footprint_results.csv
```

To run the neighbor damage visual probe:
```bash
python experiments/probe_neighbor_damage.py \
    --base_model CompVis/stable-diffusion-v1-4 \
    --method speed \
    --ckpt checkpoints/speed/few-concept/style/Van\ Gogh.pt \
    --concept vangogh
```

---

## References

- Li et al. (2026). *SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models.* ICLR 2026.
- Gandikota et al. (2023). *Erasing Concepts from Diffusion Models.* ICCV 2023.
