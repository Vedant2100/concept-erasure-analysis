import os
import json
import argparse
import torch
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(args):
    print(f"Loading pipeline: method={args.method}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    ).to(DEVICE)

    if args.method == "baseline":
        pass

    elif args.method == "speed":
        if not args.ckpt or not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"SPEED checkpoint not found: {args.ckpt}")
        print(f"  Applying SPEED weights from {args.ckpt} ...")
        state_dict = torch.load(args.ckpt, map_location="cpu")
        pipe.unet.load_state_dict(state_dict, strict=False)

    elif args.method == "esd_x":
        # Full UNet state dict. Source:
        # https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt
        if not args.ckpt or not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"ESD-x checkpoint not found: {args.ckpt}")
        print(f"  Applying ESD-x UNet weights from {args.ckpt} ...")
        state_dict = torch.load(args.ckpt, map_location="cpu")
        pipe.unet.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_probe(args):
    with open(args.prompts_file) as f:
        all_prompts = json.load(f)

    concept_prompts = all_prompts[args.concept]
    all_items = (
        [(p["id"], p["prompt"], "in_retain_set")    for p in concept_prompts["in_retain_set"]]
        + [(p["id"], p["prompt"], "not_in_retain_set") for p in concept_prompts["not_in_retain_set"]]
        + [(p["id"], p["prompt"], "unrelated")         for p in concept_prompts["unrelated"]]
    )

    pipe = load_pipeline(args)

    for prompt_id, prompt, category in all_items:
        save_dir = os.path.join(args.out_dir, args.method, category, prompt_id)
        os.makedirs(save_dir, exist_ok=True)
        print(f"  [{category}] {prompt_id}: {prompt}")
        for seed in args.seeds:
            out_path = os.path.join(save_dir, f"seed{seed}.png")
            if os.path.exists(out_path):
                print(f"    seed{seed}: already exists, skipping")
                continue
            gen = torch.Generator(DEVICE).manual_seed(seed)
            image = pipe(
                prompt,
                num_inference_steps=50,
                generator=gen,
                guidance_scale=7.5,
            ).images[0]
            image.save(out_path)

    print(f"Done. Results saved to {args.out_dir}/{args.method}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probe semantic neighbor collateral damage across erasure methods."
    )
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument(
        "--method", choices=["baseline", "speed", "esd_x"], required=True,
        help="Which model to probe. esd_x requires --ckpt pointing to "
             "diffusers-VanGogh-ESDx1-UNET.pt from erasing.baulab.info."
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="Path to .pt checkpoint file (SPEED or ESD-x UNet weights)."
    )
    parser.add_argument(
        "--concept", choices=["vangogh", "snoopy"], required=True,
        help="Which concept's erased model to probe."
    )
    parser.add_argument(
        "--prompts_file", default="experiments/neighbor_prompts.json",
        help="Path to JSON file defining prompts per concept."
    )
    parser.add_argument(
        "--out_dir", default="results/neighbor_damage",
        help="Root output directory. Images saved to out_dir/method/category/prompt_id/seedN.png"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Fixed seeds for reproducible generation."
    )
    args = parser.parse_args()
    run_probe(args)
