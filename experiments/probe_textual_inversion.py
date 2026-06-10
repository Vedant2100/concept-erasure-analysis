import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append(os.path.abspath("SPEED_repo"))
from src.template import template_dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_reference_images(base_model_id, prompt, num_images=5):
    print(f"Generating {num_images} reference images from base model...")
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    images = []
    for i in range(num_images):
        gen = torch.Generator(DEVICE).manual_seed(i)
        img = pipe(prompt, num_inference_steps=50, generator=gen).images[0]
        images.append(img)
    
    # Save them just for inspection
    os.makedirs("results/references", exist_ok=True)
    for i, img in enumerate(images):
        img.save(f"results/references/{prompt.replace(' ', '_')}_{i}.png")
        
    del pipe
    torch.cuda.empty_cache()
    return images

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
            if ckpt_path and os.path.exists(ckpt_path):
                print(f"Applying SPEED weights from {ckpt_path}...")
                pipe.unet.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        elif method == "mace":
            if ckpt_path and os.path.exists(ckpt_path):
                print(f"Applying MACE LoRA from {ckpt_path}...")
                pipe.unet.load_attn_procs(ckpt_path)
        else:
            raise ValueError(f"Unknown method {method}")
            
    pipe.set_progress_bar_config(disable=True)
    return pipe

def sample_images(pipe, prompts, token_str, save_dir, seeds=[0, 1, 2, 3]):
    os.makedirs(save_dir, exist_ok=True)
    for prompt in prompts:
        # Inject the learned token into the template
        formatted_prompt = prompt.format(token_str)
        for seed in seeds:
            gen = torch.Generator(DEVICE).manual_seed(seed)
            img = pipe(formatted_prompt, num_inference_steps=50, generator=gen).images[0]
            
            safe_prompt = formatted_prompt.replace(" ", "_").replace("/", "").replace(".", "")[:50]
            img.save(os.path.join(save_dir, f"{safe_prompt}_seed{seed}.png"))

def train_textual_inversion(args):
    # 1. Generate reference images using baseline
    reference_images = generate_reference_images(args.base_model, args.reference_prompt, num_images=5)
    
    # Image transforms for TI
    transform = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomCrop(512),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    pixel_values = torch.stack([transform(img) for img in reference_images]).to(DEVICE, dtype=torch.float16)

    # Pre-generate un-erased baseline for LPIPS comparison
    print("Generating un-erased baseline images for LPIPS comparison...")
    base_pipe = load_pipeline(args.base_model, "baseline", None)
    templates = template_dict[args.template_type][:5]
    sample_images(base_pipe, templates, args.anchor_concept, os.path.join(args.out_dir, "baseline"), seeds=[0, 1, 2, 3])
    del base_pipe
    torch.cuda.empty_cache()

    # 2. Load erased model
    pipe = load_pipeline(args.base_model, args.method, args.ckpt_path)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    vae = pipe.vae
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    
    # 3. Add new concept token to tokenizer
    num_added_tokens = tokenizer.add_tokens(args.learned_token)
    token_id = tokenizer.convert_tokens_to_ids(args.learned_token)
    
    # Resize token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Initialize the new token with the anchor concept's embedding
    anchor_token_ids = tokenizer.encode(args.anchor_concept, add_special_tokens=False)
    anchor_embedding = text_encoder.get_input_embeddings().weight.data[anchor_token_ids[0]]
    text_encoder.get_input_embeddings().weight.data[token_id] = anchor_embedding.clone()

    # Freeze everything except the new token embedding
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=args.learning_rate)
    
    # Get templates for the target concept
    templates = template_dict[args.template_type][:5] # take first 5 templates
    
    # 4. Main Training Loop over the budget grid
    budgets = sorted(args.budget_grid)
    current_step = 0
    
    for next_budget in budgets:
        steps_to_train = next_budget - current_step
        
        if steps_to_train > 0:
            print(f"\n--- Training from step {current_step} to {next_budget} ---")
            text_encoder.train()
            
            progress_bar = tqdm(total=steps_to_train)
            for _ in range(steps_to_train):
                # Sample random template
                template = templates[torch.randint(0, len(templates), (1,)).item()]
                prompt = template.format(args.learned_token)
                
                # Tokenize and encode
                text_inputs = tokenizer(
                    prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                ).to(DEVICE)
                
                # Sample batch of images
                batch_idx = torch.randint(0, len(pixel_values), (args.batch_size,))
                b_images = pixel_values[batch_idx]
                
                with torch.no_grad():
                    latents = vae.encode(b_images).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                optimizer.zero_grad()
                loss.backward()
                
                # Ensure only the new token is updated
                grads = text_encoder.get_input_embeddings().weight.grad
                index_no_updates = torch.arange(len(tokenizer)) != token_id
                grads.data[index_no_updates, :] = 0
                
                optimizer.step()
                
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.close()
            
        current_step = next_budget
        
        # 5. Evaluate and Sample at this budget
        print(f"\nSampling for budget {current_step}...")
        save_dir = os.path.join(args.out_dir, f"budget_{current_step}")
        pipe.text_encoder = text_encoder # ensure updated encoder is used
        sample_images(pipe, templates, args.learned_token, save_dir, seeds=[0, 1, 2, 3])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--method", type=str, choices=["baseline", "speed", "esd", "mace"], required=True)
    parser.add_argument("--ckpt_path", type=str, help="Path to weights (SPEED .pt, MACE lora dir, or ESD hf repo)")
    parser.add_argument("--reference_prompt", type=str, required=True, help="Prompt to generate reference images (e.g. 'a photo of Snoopy')")
    parser.add_argument("--learned_token", type=str, required=True, help="Token to optimize (e.g. '<snoopy>')")
    parser.add_argument("--anchor_concept", type=str, required=True, help="Concept to initialize the token (e.g. 'dog' or 'art')")
    parser.add_argument("--template_type", type=str, choices=["instance", "style", "celebrity"], required=True)
    parser.add_argument("--budget_grid", type=int, nargs="+", default=[0, 50, 200, 500, 1000])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--esd_model_path", type=str, default=None, help="Local path to ESD model")
    
    args = parser.parse_args()
    train_textual_inversion(args)
