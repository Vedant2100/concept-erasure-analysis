import os
import argparse
import torch
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline(base_model_id, method, ckpt_path, esd_model_path=None):
    print(f"Loading pipeline for {method}...")
    
    if method == "esd":
        model_path = esd_model_path if esd_model_path else ckpt_path
        if not model_path:
            raise ValueError("ESD requires a ckpt_path or esd_model_path")
        
        if model_path.endswith(".pt"):
            pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(DEVICE)
            print(f"Applying ESD U-Net weights from {model_path}...")
            pipe.unet.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(DEVICE)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(DEVICE)
        
        if method == "baseline":
            pass
        elif method == "speed":
            if not ckpt_path or not os.path.exists(ckpt_path):
                raise ValueError("SPEED requires a valid ckpt_path")
            print(f"Applying SPEED weights from {ckpt_path}...")
            pipe.unet.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        elif method == "mace":
            # MACE uses LoRA adapters in diffusers format
            if not ckpt_path or not os.path.exists(ckpt_path):
                raise ValueError("MACE requires a valid ckpt_path to the LoRA weights")
            print(f"Applying MACE LoRA from {ckpt_path}...")
            pipe.unet.load_attn_procs(ckpt_path)
        else:
            raise ValueError(f"Unknown method {method}")
            
    pipe.set_progress_bar_config(disable=True)
    return pipe

def run_compositional_probe(args):
    pipe = load_pipeline(args.base_model, args.method, args.ckpt_path)
    os.makedirs(args.out_dir, exist_ok=True)
    
    prompts = {
        "direct": args.prompt_direct,
        "synonym": args.prompt_synonym,
        "compositional": args.prompt_compositional
    }
    
    for p_type, prompt in prompts.items():
        if not prompt:
            continue
        print(f"Generating {p_type}: {prompt}")
        for seed in args.seeds:
            gen = torch.Generator(DEVICE).manual_seed(seed)
            img = pipe(prompt, num_inference_steps=50, generator=gen).images[0]
            img.save(os.path.join(args.out_dir, f"{args.method}_{p_type}_seed{seed}.png"))
    
    print(f"Done! Results saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--method", type=str, choices=["baseline", "speed", "esd", "mace"], required=True)
    parser.add_argument("--ckpt_path", type=str, help="Path to weights (SPEED .pt, MACE lora dir, or ESD hf repo)")
    parser.add_argument("--prompt_direct", type=str, required=True)
    parser.add_argument("--prompt_synonym", type=str, required=True)
    parser.add_argument("--prompt_compositional", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--out_dir", type=str, required=True)
    
    args = parser.parse_args()
    run_compositional_probe(args)
