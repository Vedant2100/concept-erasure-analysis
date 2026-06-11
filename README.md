# Erasing Concepts from Diffusion Models — Project Analysis

**Course:** EE243  
**Student:** Vedant  
**Frontier paper:** SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models (Li et al., ICLR 2026)

🌐 **[Project Webpage](https://vedant2100.github.io/concept-erasure-analysis)**

---

## Overview

This repository hosts a research analysis tracing the lineage from ESD (Gandikota et al., 2023) to SPEED (Li et al., ICLR 2026), evaluating training-free concept erasure in text-to-image diffusion models.

To maintain a clean master branch, all experimental code, scripts, SLURM configs, logs, and generated visual results are isolated onto dedicated development branches.

---

## Branch Mapping

* **`main`**: The clean homepage (`index.html` and `blog.css`), documenting only the research lineage diagram, strengths, and TL;DR. Contains no experimental scripts, assets, or results.
* **`all-experiments`**: Contains the complete project workspace, scripts, and results for all three experiments:
  1. **Textual Inversion Recovery Probe** — Evaluates if style/instance erasures can be "re-learned" with few-shot optimization.
  2. **Compositional Prompt Evasion Probe** — Evaluates whether synonym or detailed composition bypasses concept erasures.
  3. **Semantic Neighbor Collateral Damage Probe** — Evaluates visual style suppression in semantically adjacent artists outside the retain set.
* **`experiment-3`**: Dedicated branch focusing strictly on **Experiment 3** (Semantic Neighbor Collateral Damage), containing the mathematical footprint analysis and neighbor damage visual probes.

To run scripts or view output images, checkout the respective experiment branch:
```bash
git checkout all-experiments
# or
git checkout experiment-3
```

---

## References

- Li et al. (2026). *SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models.* ICLR 2026.
- Gandikota et al. (2023). *Erasing Concepts from Diffusion Models.* ICCV 2023.
