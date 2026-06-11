# Erasing Concepts from Diffusion Models — Project Analysis

**Course:** EE243  
**Student:** Vedant  
**Frontier paper:** SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models (Gupta et al., ICLR 2026)

🌐 **[Project Webpage](https://vedant2100.github.io/concept-erasure-analysis)**  
▶️ **[Video Walkthrough](#)** *(link added after recording)*

---

## Overview

This project traces the research lineage from ESD (Gandikota et al., 2023) through to SPEED (Gupta et al., ICLR 2026), identifies SPEED as the current open-source frontier for training-free concept erasure in diffusion models, and then empirically exposes its limitations through three targeted bottleneck probes:

1. **Textual Inversion Recovery** — evaluating if erased concepts can be "re-learned" with few-shot optimization, proving that stylistic erasures under SPEED are merely lexical, not visual.
2. **Compositional Prompt Evasion** — demonstrating that erasing a concept by its canonical name does not prevent generation via synonyms or compositional descriptions.
3. **Retain-Set Horizon (Semantic Collateral Damage)** — investigating whether the null-space projection safely preserves semantic neighbors outside of the explicit retain set (SPEED) compared to unconstrained gradient updates (ESD-x).

All experiments use independently written evaluation code in `diffusers` testing the official released checkpoints.

---

## Repo Structure

```
concept-erasure-analysis/
├── index.html                  ← GitHub Pages project webpage
├── README.md
├── requirements.txt
├── blog.css                    ← Styling for the webpage
├── experiments/
│   ├── probe_ti_recovery.py    ← Experiment 1: Textual Inversion recovery
│   ├── probe_compositional.py  ← Experiment 2: Compositional prompt evasion
│   └── probe_neighbor_damage.py← Experiment 3: Semantic neighbor collateral damage
└── results/                    ← Generated evaluation outputs
```

---

## References

- Gupta et al. (2026). *SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models.* ICLR 2026.
- Gandikota et al. (2023). *Erasing Concepts from Diffusion Models.* ICCV 2023.
