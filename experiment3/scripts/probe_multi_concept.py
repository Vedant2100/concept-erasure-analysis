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
    # Retain-set collapse canaries — never erased, always in R_refine
    {"id": "gauguin",   "prompt": "a painting in the style of Paul Gauguin",    "category": "retain_canary"},
    {"id": "seurat",    "prompt": "a painting in the style of Georges Seurat",  "category": "retain_canary"},
    {"id": "pissarro",  "prompt": "a painting in the style of Camille Pissarro","category": "retain_canary"},
    # Repeat Rysselberghe as non-retain bridge artist
    {"id": "rysselberghe","prompt": "a painting in the style of Theo van Rysselberghe","category": "non_retain"},
]

def load_pipeline(base_model, ckpt_path):
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
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
            image.save(out_path)
            print(f"    seed{seed}: saved")

    print(f"Done. Results in {args.out_dir}/{args.method}/")


if __name__ == "__main__":
    main()
