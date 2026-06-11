"""
probe_rank_saturation.py — Experiment 3.4: does SPEED's null-space collapse when a
GROWING, CONCENTRATED cluster of same-movement concepts is erased at once?

Motivation (SPEED's own admitted limitation, paper Sec. 4 / App. E):
  "As R increases, C0 C0^T gradually reaches full rank, its null space narrows and
   reduces to the trivial null space {0}."
SPEED evaluated 100 DIVERSE celebrities and 5 painters erased INDIVIDUALLY. It never
erased many artists of ONE movement and measured a held-out artist of that movement.
Mass Concept Erasure (arXiv 2601.03305) shows that for other methods, erasing many
semantically-similar subtypes degrades the shared supertype via feature interference.
This probe tests whether SPEED's null-space resists that, and if so up to what N.

Design: erase a nested, concentrated cluster of impressionist/post-impressionist
painters (N = 5, 10, 20, 40 — built by slurm_rank_saturation.sh). Then measure, on
NEVER-erased prompts, how much SPEED has drifted from baseline as N grows:
  * canaries     — Gauguin / Seurat / Pissarro: in the retain set, impressionist,
                   held out. The exposed members.
  * supertype    — "an impressionist oil painting" etc.: the shared parent capability.
  * controls_far — Rembrandt / Hokusai: retained but style-distant. Must stay flat.
  * erased_check — Renoir (erased at every N): sanity that erasure actually fires.

fp32 + safety_checker off + is_black guard: same fix as Exp 3.3 so nude-heavy
painters (Gauguin etc.) cannot blank to black and poison the CLIP-drift metric.
"""
import os
import argparse
import torch
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPTS = [
    # Sanity: this artist is erased at every N — must show high drift throughout.
    {"id": "renoir",        "prompt": "a painting in the style of Pierre-Auguste Renoir", "category": "erased_check"},
    # Held-out canaries: retained, impressionist-family, NEVER erased.
    {"id": "gauguin",       "prompt": "a painting in the style of Paul Gauguin",     "category": "canary"},
    {"id": "seurat",        "prompt": "a painting in the style of Georges Seurat",   "category": "canary"},
    {"id": "pissarro",      "prompt": "a painting in the style of Camille Pissarro", "category": "canary"},
    # Supertype capability: the shared parent concept, tied to no single erased artist.
    {"id": "impressionist", "prompt": "an impressionist oil painting of a garden",   "category": "supertype"},
    {"id": "postimpr",      "prompt": "a post-impressionist landscape painting",     "category": "supertype"},
    {"id": "pointillist",   "prompt": "a pointillist painting of a harbor",          "category": "supertype"},
    # Negative controls: retained, style-FAR. Must stay flat as N grows.
    {"id": "rembrandt",     "prompt": "a painting in the style of Rembrandt",        "category": "control_far"},
    {"id": "hokusai",       "prompt": "a painting in the style of Hokusai",          "category": "control_far"},
]

METHOD_CHOICES = ["baseline", "speed_5c", "speed_10c", "speed_20c", "speed_40c"]


def is_black(image, thresh=10.0):
    px = list(image.convert("L").getdata())
    return (sum(px) / len(px)) < thresh


def load_pipeline(base_model, ckpt_path):
    # fp32 + NSFW safety checker OFF: prevents black-frame contamination (the checker
    # blanks nude-heavy painters; fp16 can NaN the VAE). Does NOT change seeds, so the
    # paired baseline-vs-edited comparison stays valid.
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        pipe.unet.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print("  Running baseline (no checkpoint)")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--method", required=True, choices=METHOD_CHOICES)
    ap.add_argument("--ckpt", default=None,
                    help="Path to SPEED .pt checkpoint (not needed for baseline)")
    ap.add_argument("--out_dir", default="experiment3/results/rank_saturation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    if args.method != "baseline" and not args.ckpt:
        ap.error(f"--ckpt is required for method={args.method}")

    pipe = load_pipeline(args.base_model, args.ckpt)

    for item in TEST_PROMPTS:
        save_dir = os.path.join(args.out_dir, args.method, item["category"], item["id"])
        os.makedirs(save_dir, exist_ok=True)
        print(f"  [{item['category']}] {item['id']}: {item['prompt']}")
        for seed in args.seeds:
            out_path = os.path.join(save_dir, f"seed{seed}.png")
            if os.path.exists(out_path):
                print(f"    seed{seed}: exists, skipping")
                continue
            gen = torch.Generator(DEVICE).manual_seed(seed)
            image = pipe(item["prompt"], num_inference_steps=50,
                         generator=gen, guidance_scale=7.5).images[0]
            if is_black(image):
                print(f"    seed{seed}: WARNING — black frame after fp32+no-safety-checker")
                with open(os.path.join(args.out_dir, "corrupt_frames.txt"), "a") as cf:
                    cf.write(f"{args.method}/{item['category']}/{item['id']}/seed{seed}.png\n")
            image.save(out_path)
            print(f"    seed{seed}: saved")

    print(f"Done. Results in {args.out_dir}/{args.method}/")


if __name__ == "__main__":
    main()
