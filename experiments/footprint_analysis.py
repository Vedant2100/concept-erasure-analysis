"""
footprint_analysis.py

For each candidate prompt, measures how much SPEED's edit displaces its
key-projection representations relative to the erased concept ("Van Gogh").

    relative_footprint(p) = ||ΔW_K @ embed(p)||_F / ||ΔW_K @ embed("Van Gogh")||_F

Prompts with high relative footprint AND not in R_refine are the empirical
retain-set horizon — concepts SPEED's edit touches without any protection guarantee.

Does NOT generate images. CPU-only (no GPU needed). Outputs a CSV sorted by
SPEED footprint descending so you know which prompts to probe next.

Usage:
    python experiments/footprint_analysis.py \
        --speed_ckpt checkpoints/speed/few-concept/style/Van\ Gogh.pt \
        --base_model CompVis/stable-diffusion-v1-4 \
        --retain_set SPEED_repo/data/style.csv \
        --out footprint_results.csv
"""

import argparse
import csv
import os
import torch
from diffusers import StableDiffusionPipeline

# ---------------------------------------------------------------------------
# Candidate prompts
# All formatted as "a painting in the style of X" for consistency with probes.
# in_retain_set = True means we confirmed the concept IS in style.csv (controls).
# in_retain_set = False means confirmed ABSENT from style.csv (candidates).
# in_retain_set = None means unverified — script will check at runtime.
# ---------------------------------------------------------------------------
CANDIDATES = [
    # --- Controls: confirmed IN R_refine (expect ~0 footprint) ---
    ("Monet",       "a painting in the style of Claude Monet",          True),
    ("Cézanne",     "a painting in the style of Paul Cézanne",          True),
    ("Gauguin",     "a painting in the style of Paul Gauguin",          True),
    ("Seurat",      "a painting in the style of Georges Seurat",        True),
    ("Pissarro",    "a painting in the style of Camille Pissarro",      True),
    ("Guillaumin",  "a painting in the style of Armand Guillaumin",     True),
    ("Toulouse-Lautrec", "a painting in the style of Henri de Toulouse-Lautrec", True),
    ("Émile Bernard", "a painting in the style of Émile Bernard",       True),
    ("Cross",       "a painting in the style of Henri-Edmond Cross",    True),
    ("Angrand",     "a painting in the style of Charles Angrand",       True),
    ("Maximilien Luce", "a painting in the style of Maximilien Luce",   True),
    ("Daubigny",    "a painting in the style of Charles-François Daubigny", True),
    ("Anton Mauve", "a painting in the style of Anton Mauve",           True),
    ("Corot",       "a painting in the style of Camille Corot",         True),
    ("Courbet",     "a painting in the style of Gustave Courbet",       True),

    # --- Candidates: confirmed NOT in R_refine (the retain-set horizon targets) ---
    # Direct influences on Van Gogh or stylistic neighbors not in retain set
    ("Rysselberghe", "a painting in the style of Theo van Rysselberghe", False),
    ("Monticelli",  "a painting in the style of Adolphe Monticelli",    False),
    ("van Rappard", "a painting in the style of Anton van Rappard",     False),
    ("Breitner",    "a painting in the style of George Hendrik Breitner", False),
    ("Toorop",      "a painting in the style of Jan Toorop",            False),
    ("Jongkind",    "a painting in the style of Johan Barthold Jongkind", False),

    # Genre/style descriptors (not artist names)
    ("post-impressionist", "a post-impressionist landscape painting",   False),
    ("expressionist",      "an expressionist painting with bold colors", False),
    ("fauvism",            "a Fauvist painting with vivid flat color",  False),
    ("pointillist",        "a pointillist painting made of small dots", False),
    ("impasto",            "a painting with thick impasto brushstrokes and swirling bold colors", False),
    ("Starry Night desc",  "Starry Night painting with swirling sky over a village", False),
    ("Dutch Golden Age",   "a Dutch Golden Age landscape painting",     False),
    ("plein air",          "a plein air landscape in oil paint",        False),

    # Unrelated controls (expect ~0 footprint for both methods)
    ("mountain photo",     "a realistic photograph of a mountain at sunrise", False),
    ("portrait photo",     "a photorealistic portrait of a person",     False),
    ("anime",              "an anime character illustration",            False),
]


def get_delta_W(speed_sd, baseline_sd):
    """Return {layer_key: delta_tensor} for cross-attention K and V projections."""
    delta = {}
    for key in speed_sd:
        if key in baseline_sd and "attn2" in key and ("to_k" in key or "to_v" in key):
            delta[key] = (speed_sd[key] - baseline_sd[key]).float()
    return delta


def compute_footprint(delta_W, text_embed):
    """
    text_embed: [seq_len, d_text]  (full token sequence, not pooled)
    Returns total Frobenius norm of ΔW @ embed.T summed across all layers.
    """
    total = 0.0
    for dW in delta_W.values():
        # dW: [d_k, d_text], text_embed.T: [d_text, seq_len]
        displaced = dW @ text_embed.T.float()   # [d_k, seq_len]
        total += displaced.norm().item()
    return total


def encode_prompt(prompt, pipe):
    """Return the CLIP text encoder's last hidden state for a prompt: [seq_len, d_text]."""
    tokens = pipe.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        padding="max_length",
    )
    with torch.no_grad():
        output = pipe.text_encoder(**tokens)
    return output.last_hidden_state.squeeze(0)   # [77, 768]


def check_retain_set(retain_set_path):
    """Return a set of lowercase concept names from style.csv."""
    concepts = set()
    with open(retain_set_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            concepts.add(row["concept"].strip().lower())
    return concepts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed_ckpt", required=True)
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--retain_set", default="SPEED_repo/data/style.csv")
    parser.add_argument("--out", default="footprint_results.csv")
    args = parser.parse_args()

    print("Loading pipeline (baseline)...")
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float32)
    pipe.text_encoder.eval()
    baseline_sd = pipe.unet.state_dict()

    print(f"Loading SPEED checkpoint from {args.speed_ckpt} ...")
    speed_sd = torch.load(args.speed_ckpt, map_location="cpu")

    delta_W = get_delta_W(speed_sd, baseline_sd)
    print(f"  Found {len(delta_W)} modified cross-attention K/V layers.")

    if not delta_W:
        raise RuntimeError(
            "No cross-attention K/V delta found. Check checkpoint format — "
            "keys should contain 'attn2' and 'to_k'/'to_v'."
        )

    # Reference footprint: the erased concept itself
    reference_prompts = [
        "a painting in the style of Van Gogh",
        "a painting in the style of Vincent van Gogh",
    ]
    ref_embed = encode_prompt(reference_prompts[0], pipe)
    ref_footprint = compute_footprint(delta_W, ref_embed)
    print(f"  Van Gogh reference footprint: {ref_footprint:.4f}")

    retain_concepts = set()
    if os.path.exists(args.retain_set):
        retain_concepts = check_retain_set(args.retain_set)
        print(f"  Loaded {len(retain_concepts)} retain-set concepts from {args.retain_set}")

    rows = []
    for label, prompt, in_retain_known in CANDIDATES:
        embed = encode_prompt(prompt, pipe)
        fp = compute_footprint(delta_W, embed)
        rel = fp / ref_footprint if ref_footprint > 0 else 0.0

        # Runtime retain-set check: extract the artist name from the prompt
        # (rough heuristic: last few words after "style of")
        if in_retain_known is None:
            extract = prompt.split("style of")[-1].strip().lower() if "style of" in prompt else prompt.lower()
            in_retain_known = extract in retain_concepts

        rows.append({
            "label": label,
            "prompt": prompt,
            "in_retain_set": in_retain_known,
            "footprint": round(fp, 6),
            "relative_footprint": round(rel, 6),
        })
        print(f"  {label:30s}  rel={rel:.4f}  retain={in_retain_known}")

    rows.sort(key=lambda r: r["relative_footprint"], reverse=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "prompt", "in_retain_set", "footprint", "relative_footprint"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {args.out}")
    print("\nTop candidates NOT in retain set (the retain-set horizon):")
    horizon = [r for r in rows if not r["in_retain_set"]][:10]
    for r in horizon:
        flag = " <-- PROBE THESE" if r["relative_footprint"] > 0.05 else ""
        print(f"  {r['label']:30s}  rel={r['relative_footprint']:.4f}{flag}")


if __name__ == "__main__":
    main()
