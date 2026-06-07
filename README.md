# Erasing Concepts from Diffusion Models — Project Analysis

**Course:** EE243  
**Student:** Vedant   
**Frontier paper:** MACE — Mass Concept Erasure in Text-to-Image Diffusion Models (Lu et al., CVPR 2024)

🌐 **[Project Webpage](https://vedant2100.github.io/concept-erasure-analysis)**  
▶️ **[Video Walkthrough](#)** *(link added after recording)*

---

## Overview

This project traces the research lineage from ESD (Gandikota et al., 2023) through to MACE (Lu et al., CVPR 2024), identifies MACE as the current open-source frontier for concept erasure in diffusion models, and then empirically exposes two concrete failure modes:

1. **Compositional prompt evasion** — erasing a concept by name does not prevent generation via synonyms, hypernyms, or compositional scene descriptions.
2. **Semantic collateral damage** — erasing one concept degrades generation quality for semantically neighboring concepts that were never targeted.

No training or fine-tuning is performed. All experiments use MACE's released checkpoints.

---

## Repo Structure

```
concept-erasure-analysis/
├── index.html                  ← GitHub Pages project webpage
├── README.md
├── requirements.txt
├── experiments/
│   ├── setup_mace.sh           ← clone & install MACE
│   ├── exp1_compositional.py   ← Experiment 1: evasion via synonyms/composition
│   └── exp2_collateral.py      ← Experiment 2: semantic neighborhood damage
└── results/
    ├── strengths/              ← success case images (add after running)
    ├── exp1/                   ← compositional evasion outputs
    └── exp2/                   ← collateral damage outputs
```

---

## Setup

### 1. Clone this repo
```bash
git clone https://github.com/vedant2100/concept-erasure-analysis.git
cd concept-erasure-analysis
```

### 2. Install MACE and dependencies
```bash
bash experiments/setup_mace.sh
```

This clones the official MACE repo, installs dependencies, and downloads the erased checkpoints from HuggingFace.

### 3. Run experiments
```bash
# Experiment 1 — compositional evasion
python experiments/exp1_compositional.py

# Experiment 2 — semantic collateral damage
python experiments/exp2_collateral.py
```

Outputs are saved to `results/exp1/` and `results/exp2/` respectively.

---

## Lineage Summary

| Paper | Year | Key contribution |
|-------|------|-----------------|
| Safe Latent Diffusion | 2022 | Inference-time steering (bypassable) |
| ESD | Mar 2023 | Fine-tune weights using negative guidance — first weight-editing approach |
| UCE | Aug 2023 | Closed-form weight edit, multi-concept in one shot |
| Concept Ablation / SPM | 2023 | Adapter-based tuning for better isolation |
| **MACE** ★ | CVPR 2024 | Scales to 100 concepts; LoRA + attention refinement; best generality/specificity balance |
| AdvUnlearn / ANT / SCORE | 2024–25 | Adversarial robustness, trajectory steering |

---

## References

- Lu et al. (2024). *MACE: Mass Concept Erasure in Diffusion Models.* CVPR 2024. [arxiv](https://arxiv.org/abs/2403.06135)
- Gandikota et al. (2023). *Erasing Concepts from Diffusion Models.* ICCV 2023. [arxiv](https://arxiv.org/abs/2303.07345)
- Gandikota et al. (2023). *Unified Concept Editing in Diffusion Models.* WACV 2024. [arxiv](https://arxiv.org/abs/2308.14761)
- MACE official code: [github.com/Shilin-LU/MACE](https://github.com/Shilin-LU/MACE)
