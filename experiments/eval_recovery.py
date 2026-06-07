import os
import argparse
import glob
import json
import torch
import lpips
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to probe results e.g. results/probe_ti/snoopy")
    parser.add_argument("--target_concept", type=str, required=True)
    parser.add_argument("--anchor_concept", type=str, required=True)
    args = parser.parse_args()

    # Init models
    print("Loading CLIP and LPIPS models...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE)
    
    # Text features for recovery classification
    text_inputs = clip_processor(text=[args.target_concept, args.anchor_concept], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    budgets = [d for d in os.listdir(args.results_dir) if d.startswith("budget_")]
    budgets.sort(key=lambda x: int(x.split('_')[1]))
    
    baseline_dir = os.path.join(args.results_dir, "baseline")
    if not os.path.exists(baseline_dir):
        print(f"Baseline dir not found at {baseline_dir}, skipping.")
        return

    results = []

    for budget in budgets:
        budget_dir = os.path.join(args.results_dir, budget)
        images = glob.glob(os.path.join(budget_dir, "*.png"))
        
        if len(images) == 0:
            continue
            
        lpips_scores = []
        clip_target_scores = []
        recovery_count = 0
        
        for img_path in images:
            img_name = os.path.basename(img_path)
            # Find corresponding baseline image
            # Since baseline uses anchor concept in the prompt, the filename will be different.
            # Filenames are like "a_photo_of_<snoopy>_seed0.png"
            # Baseline filenames: "a_photo_of_dog_seed0.png"
            # So we can match by replacing "<snoopy>" or "<vangogh>" with the anchor.
            # A safer way is to just match the seed suffix.
            seed_suffix = img_name.split("_seed")[-1]
            baseline_match = glob.glob(os.path.join(baseline_dir, f"*_seed{seed_suffix}"))
            
            img_pil = load_image(img_path)
            
            # CLIP features
            img_inputs = clip_processor(images=img_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                img_features = clip_model.get_image_features(**img_inputs)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                
                # similarities: shape (1, 2)
                sims = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                
                target_sim = sims[0, 0].item()
                anchor_sim = sims[0, 1].item()
                
                if target_sim > anchor_sim:
                    recovery_count += 1
                
                # Absolute CS-target (not softmaxed)
                raw_sim = (img_features @ text_features.T)[0, 0].item()
                clip_target_scores.append(raw_sim)

            # LPIPS
            if baseline_match:
                baseline_pil = load_image(baseline_match[0])
                # Convert PIL to [-1, 1] tensor for LPIPS
                tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                img_t = tr(img_pil).unsqueeze(0).to(DEVICE)
                base_t = tr(baseline_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    d = lpips_fn(img_t, base_t)
                    lpips_scores.append(d.item())

        res = {
            "budget": int(budget.split('_')[1]),
            "recovery_rate": recovery_count / len(images),
            "cs_target": sum(clip_target_scores) / len(clip_target_scores),
            "lpips": sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0.0
        }
        results.append(res)
        print(f"{budget}: {res}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.results_dir, "evaluation_metrics.csv"), index=False)
    print(f"Saved evaluation metrics to {args.results_dir}/evaluation_metrics.csv")

if __name__ == "__main__":
    main()
