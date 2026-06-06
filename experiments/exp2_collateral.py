"""
Experiment 2: Semantic Collateral Damage
=========================================
Hypothesis: When MACE erases concept C, the damage spreads to
            semantically neighbouring concepts that were never targeted.

Protocol:
  1. Erase "church" using MACE (a known hard case from the ESD paper).
  2. Generate a semantic neighbourhood of prompts ranging from
     near-synonyms → visual neighbours → unrelated.
  3. Measure LPIPS distance between baseline and erased outputs
     (higher LPIPS = more change = more collateral damage).
  4. Show qualitative grids side by side.

Outputs images + an LPIPS table to results/exp2/.
"""

import os, json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

# LPIPS for perceptual distance
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[WARN] lpips not installed. Install with: pip install lpips")
    print("       LPIPS scores will be skipped; images will still be saved.")

# ── CONFIG ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/exp2")
SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASELINE_CKPT = "CompVis/stable-diffusion-v1-4"
# MACE checkpoint with "church" erased — run MACE inference script to generate,
# or replace with a pre-erased HF checkpoint if available.
MACE_CHURCH_CKPT = "checkpoints/mace_church"

# Semantic neighbourhood — ordered from closest to most distant
NEIGHBOURHOOD = {
    "TARGET":          "a church",
    "near_syn_1":      "a cathedral",
    "near_syn_2":      "a chapel",
    "visual_nbr_1":    "a mosque",
    "visual_nbr_2":    "a temple",
    "visual_nbr_3":    "a tall stone building with a tower",
    "visual_nbr_4":    "a stone archway",
    "unrelated_1":     "a car",
    "unrelated_2":     "a dog sitting in a park",
    "unrelated_3":     "a bowl of fruit on a table",
}

# Number of seeds to average LPIPS over
NUM_SEEDS = 3
# ────────────────────────────────────────────────────────────────────────


def load_pipeline(ckpt_path: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, prompt: str, seed: int) -> Image.Image:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = pipe(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=IMG_SIZE, width=IMG_SIZE,
        generator=generator,
    )
    return result.images[0]


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to LPIPS-compatible tensor [-1, 1]."""
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 127.5 - 1.0
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def compute_lpips(loss_fn, img_base: Image.Image, img_mace: Image.Image) -> float:
    t_base = pil_to_tensor(img_base)
    t_mace = pil_to_tensor(img_mace)
    with torch.no_grad():
        dist = loss_fn(t_base, t_mace).item()
    return dist


def make_comparison_strip(imgs_base, imgs_mace, label: str, out_path: Path):
    """Save a horizontal strip: [baseline | erased] for each seed."""
    n = len(imgs_base)
    W, H = IMG_SIZE, IMG_SIZE
    strip = Image.new("RGB", (W * 2 * n, H + 30), (245, 244, 239))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(strip)
    for i, (b, m) in enumerate(zip(imgs_base, imgs_mace)):
        strip.paste(b.resize((W, H)), (i * 2 * W, 30))
        strip.paste(m.resize((W, H)), (i * 2 * W + W, 30))
    draw.text((4, 6), f"LEFT: baseline  |  RIGHT: MACE erased  |  {label}", fill=(60, 58, 53))
    strip.save(out_path)


def main():
    if not Path(MACE_CHURCH_CKPT).exists():
        print(f"[ERROR] MACE church checkpoint not found at '{MACE_CHURCH_CKPT}'.")
        print("  Run: python MACE/generate_erased.py --concept 'church' --output checkpoints/mace_church")
        print("  Or adjust MACE_CHURCH_CKPT in this script.")
        raise SystemExit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup LPIPS
    loss_fn = None
    if LPIPS_AVAILABLE:
        loss_fn = lpips.LPIPS(net="alex").to(DEVICE)
        print("LPIPS (AlexNet) loaded.")

    # Load pipelines
    print("Loading baseline SD 1.4...")
    base_pipe = load_pipeline(BASELINE_CKPT)
    print(f"Loading MACE (church erased) from {MACE_CHURCH_CKPT}...")
    mace_pipe = load_pipeline(MACE_CHURCH_CKPT)

    results = {}

    for key, prompt in NEIGHBOURHOOD.items():
        print(f"\n[{key}] \"{prompt}\"")
        imgs_base, imgs_mace = [], []
        lpips_scores = []

        for seed in range(SEED, SEED + NUM_SEEDS):
            img_b = generate(base_pipe, prompt, seed)
            img_m = generate(mace_pipe, prompt, seed)
            imgs_base.append(img_b)
            imgs_mace.append(img_m)

            img_b.save(RESULTS_DIR / f"{key}_seed{seed}_baseline.jpg")
            img_m.save(RESULTS_DIR / f"{key}_seed{seed}_mace.jpg")

            if loss_fn is not None:
                score = compute_lpips(loss_fn, img_b, img_m)
                lpips_scores.append(score)
                print(f"  seed={seed}  LPIPS={score:.4f}")

        if lpips_scores:
            mean_lpips = float(np.mean(lpips_scores))
            std_lpips = float(np.std(lpips_scores))
        else:
            mean_lpips = std_lpips = None

        results[key] = {
            "prompt": prompt,
            "lpips_mean": mean_lpips,
            "lpips_std": std_lpips,
        }

        # Save comparison strip
        make_comparison_strip(imgs_base, imgs_mace, f"{key}: {prompt}",
                              RESULTS_DIR / f"{key}_strip.jpg")
        print(f"  Strip saved → {RESULTS_DIR / f'{key}_strip.jpg'}")

    # Save JSON results table
    with open(RESULTS_DIR / "lpips_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Prompt key':<22} {'Prompt':<40} {'LPIPS':>8}")
    print("-"*60)
    for key, v in results.items():
        score_str = f"{v['lpips_mean']:.4f} ±{v['lpips_std']:.4f}" if v["lpips_mean"] else "  N/A"
        print(f"{key:<22} {v['prompt']:<40} {score_str:>8}")
    print("="*60)
    print(f"\nAll outputs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
