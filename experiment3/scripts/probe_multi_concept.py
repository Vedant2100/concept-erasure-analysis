import os
import argparse
import torch
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prompts and which erasure scale they belong to
# erased_in: list of concept counts where this artist is actively erased
TEST_PROMPTS = [
    {"id": "vangogh",   "prompt": "a painting in the style of Van Gogh",       "category": "erased"},
    {"id": "picasso",   "prompt": "a painting in the style of Pablo Picasso",   "category": "erased_in_2c_3c"},
    {"id": "monet",     "prompt": "a painting in the style of Claude Monet",    "category": "erased_in_3c"},
    # Retain-set canaries — never erased, in R_refine, but STYLE-ADJACENT to the
    # erased impressionists. If the null-space concentrates the erase direction onto
    # the shared "impressionist" subspace, these should drift monotonically 1c->2c->3c.
    {"id": "gauguin",   "prompt": "a painting in the style of Paul Gauguin",    "category": "retain_canary"},
    {"id": "seurat",    "prompt": "a painting in the style of Georges Seurat",  "category": "retain_canary"},
    {"id": "pissarro",  "prompt": "a painting in the style of Camille Pissarro","category": "retain_canary"},
    # Retain-set NEGATIVE CONTROLS — in R_refine, but stylistically FAR from
    # impressionism. These should stay FLAT across 1c->2c->3c. If they drift as much
    # as the canaries, the effect is global noise, not concentrated null-space pressure.
    {"id": "rembrandt", "prompt": "a painting in the style of Rembrandt",       "category": "retain_control_far"},
    {"id": "hokusai",   "prompt": "a painting in the style of Hokusai",         "category": "retain_control_far"},
    # Non-retain bridge artist (carried over from 3.2)
    {"id": "rysselberghe","prompt": "a painting in the style of Theo van Rysselberghe","category": "non_retain"},
]

def is_black(image, thresh=10.0):
    """Detect blank/near-black frames (mean luminance below thresh on 0-255)."""
    px = list(image.convert("L").getdata())
    return (sum(px) / len(px)) < thresh


def load_pipeline(base_model, ckpt_path):
    # Two root causes of the first run's 21 black frames, fixed here:
    #  (1) NSFW safety checker blanks flagged outputs to solid black. It fires on
    #      nude-heavy painters (Gauguin's Tahitian women, Picasso, Rembrandt) —
    #      same seed -> same composition -> same trigger -> consistently black.
    #  (2) fp16 can NaN the VAE decode, also yielding black.
    # Disabling the safety checker + running fp32 removes both WITHOUT changing the
    # seed, so the paired baseline-vs-edited comparison stays valid (the underlying
    # latent was fine; the checker was just blanking it post-hoc). Disabling the
    # NSFW filter is standard practice in the ESD/SPEED erasure codebases.
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        pipe.unet.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print("  Running baseline (no checkpoint)")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--method", required=True,
                        choices=["baseline", "speed_1c", "speed_2c", "speed_3c", "esd_x"],
                        help="Which model config to run")
    parser.add_argument("--ckpt", default=None,
                        help="Path to .pt checkpoint (SPEED or ESD-x UNet weights; not needed for baseline)")
    parser.add_argument("--out_dir", default="experiment3/results/multi_concept",
                        help="Output root; images go to out_dir/method/category/id/seedN.png")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    if args.method not in ("baseline",) and not args.ckpt:
        parser.error(f"--ckpt is required for method={args.method}")

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
            image = pipe(
                item["prompt"],
                num_inference_steps=50,
                generator=gen,
                guidance_scale=7.5,
            ).images[0]
            if is_black(image):
                # Should no longer happen with fp32 + safety checker off. If it
                # ever does, record it loudly rather than silently shipping a
                # black frame that would poison the CLIP-drift metric again.
                print(f"    seed{seed}: WARNING — still black after fp32+no-safety-checker")
                with open(os.path.join(args.out_dir, "corrupt_frames.txt"), "a") as cf:
                    cf.write(f"{args.method}/{item['category']}/{item['id']}/seed{seed}.png\n")
            image.save(out_path)
            print(f"    seed{seed}: saved")

    print(f"Done. Results in {args.out_dir}/{args.method}/")


if __name__ == "__main__":
    main()
