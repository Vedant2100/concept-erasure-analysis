"""
analyze_paraphrase.py — Experiment 4 verdict: does SPEED's style erasure survive
paraphrasing?

For every (method, prompt) cell, computes two metrics against the BASELINE:

  1. Van-Gogh-ness (CLIP image→text):
     cos(CLIP_image(generated), CLIP_text("a painting in the style of Van Gogh"))
     High = the image still carries Van Gogh's style.

  2. CLIP image→image drift vs baseline:
     1 - cos(CLIP_image(baseline_img), CLIP_image(edited_img))
     Measures how much the edited model changed the output for that same prompt/seed.

The finding lives in the CONTRAST across prompt tiers:

  EVASION CONFIRMED (limitation):
    - Named prompts: Van-Gogh-ness drops sharply in SPEED & ESD (erasure fires).
    - Paraphrase prompts: Van-Gogh-ness stays HIGH in SPEED (erasure bypassed).
    - Bonus: if ESD also fails → both methods share the weakness.
    - Bonus: if ESD fails MORE → SPEED's null-space buys some robustness.

  ROBUST (honest negative):
    - Paraphrase Van-Gogh-ness drops as much as named → SPEED resists paraphrasing.

Black/corrupt frames are excluded and counted.
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match probe_paraphrase.py prompt IDs and categories
PROMPTS = [
    ("named_vangogh",     "named",              "Named: 'style of Van Gogh'"),
    ("named_starry",      "named",              "Named: 'Starry Night by Van Gogh'"),
    ("para_swirl",        "paraphrase_desc",    "Desc: swirling impasto night sky"),
    ("para_wheat",        "paraphrase_desc",    "Desc: golden wheat turbulent sky"),
    ("para_cafe",         "paraphrase_desc",    "Desc: cafe terrace complementary"),
    ("para_postimpr_swirl","paraphrase_style",  "Style: post-impr swirling rhythmic"),
    ("para_expressive",   "paraphrase_style",   "Style: expressionist impasto"),
    ("para_cypress",      "paraphrase_style",   "Style: cypress swirling sky"),
    ("para_emotion",      "paraphrase_abstract","Abstract: emotional turmoil"),
    ("para_movement",     "paraphrase_abstract","Abstract: sky moves with energy"),
    ("ctrl_ukiyo",        "control_far",        "Control: ukiyo-e woodblock"),
    ("ctrl_rembrandt",    "control_far",        "Control: Rembrandt chiaroscuro"),
    ("ctrl_hokusai",      "control_far",        "Control: Hokusai Mount Fuji"),
]

METHODS = ["speed_1c", "esd_x"]
VG_TEXT = "a painting in the style of Van Gogh"


def is_black(path, thresh=10.0):
    px = list(Image.open(path).convert("L").getdata())
    return (sum(px) / len(px)) < thresh


@torch.no_grad()
def clip_img_emb(model, proc, path):
    e = model.get_image_features(**proc(images=Image.open(path).convert("RGB"),
                                        return_tensors="pt").to(DEVICE))
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_txt_emb(model, proc, text):
    e = model.get_text_features(**proc(text=[text], return_tensors="pt",
                                       padding=True).to(DEVICE))
    return e / e.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiment4/results/paraphrase")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_csv", default="experiment4/results/paraphrase/evasion.csv")
    args = ap.parse_args()

    print(f"Loading CLIP: {args.clip}")
    model = CLIPModel.from_pretrained(args.clip).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(args.clip)

    # Pre-compute the text embedding for "Van Gogh style"
    vg_emb = clip_txt_emb(model, proc, VG_TEXT)

    rows = []
    n_excluded = 0

    # Header
    hdr = f"{'prompt_id':<22}{'tier':<22}"
    for m in ["baseline"] + METHODS:
        hdr += f"{'VG-ness('+m+')':<18}"
    for m in METHODS:
        hdr += f"{'drift('+m+')':<16}"
    hdr += "  n"
    print("\n" + hdr)
    print("-" * len(hdr))

    for pid, cat, role in PROMPTS:
        cell_vgness = {}
        cell_drift = {}
        cell_n = {}

        # Baseline Van-Gogh-ness
        vg_scores_base = []
        for s in args.seeds:
            bp = f"{args.root}/baseline/{cat}/{pid}/seed{s}.png"
            if not os.path.exists(bp):
                continue
            if is_black(bp):
                n_excluded += 1
                continue
            be = clip_img_emb(model, proc, bp)
            vg_scores_base.append((be * vg_emb).sum().item())
        cell_vgness["baseline"] = np.mean(vg_scores_base) if vg_scores_base else float("nan")

        # Each method
        for method in METHODS:
            vg_scores, drifts = [], []
            for s in args.seeds:
                bp = f"{args.root}/baseline/{cat}/{pid}/seed{s}.png"
                ep = f"{args.root}/{method}/{cat}/{pid}/seed{s}.png"
                if not (os.path.exists(bp) and os.path.exists(ep)):
                    continue
                if is_black(bp) or is_black(ep):
                    n_excluded += 1
                    continue
                ee = clip_img_emb(model, proc, ep)
                be = clip_img_emb(model, proc, bp)
                vg_scores.append((ee * vg_emb).sum().item())
                drifts.append(1.0 - (be * ee).sum().item())

            cell_vgness[method] = np.mean(vg_scores) if vg_scores else float("nan")
            cell_drift[method] = np.mean(drifts) if drifts else float("nan")
            cell_n[method] = len(vg_scores)

        # Print row
        line = f"{pid:<22}{cat:<22}"
        for m in ["baseline"] + METHODS:
            line += f"{cell_vgness.get(m, float('nan')):<18.4f}"
        for m in METHODS:
            line += f"{cell_drift.get(m, float('nan')):<16.4f}"
        nmin = min(cell_n.values()) if cell_n else 0
        line += f"{nmin:>4}"
        print(line)

        rows.append((pid, cat, role, cell_vgness, cell_drift, cell_n))

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        cols = ["prompt_id", "category", "role"]
        for m in ["baseline"] + METHODS:
            cols.append(f"vgness_{m}")
        for m in METHODS:
            cols += [f"drift_{m}", f"n_{m}"]
        f.write(",".join(cols) + "\n")
        for pid, cat, role, vgn, dft, ns in rows:
            vals = [pid, cat, f'"{role}"']
            for m in ["baseline"] + METHODS:
                vals.append(f"{vgn.get(m, float('nan')):.5f}")
            for m in METHODS:
                vals.append(f"{dft.get(m, float('nan')):.5f}")
                vals.append(str(ns.get(m, 0)))
            f.write(",".join(vals) + "\n")

    print(f"\nWrote {args.out_csv}")
    if n_excluded:
        print(f"NOTE: excluded {n_excluded} seed(s) with corrupt/black frames.")

    # --- Verdict guide ---
    print("\n=== VERDICT GUIDE ===")
    print("EVASION CONFIRMED (limitation):")
    print("  Named VG-ness drops sharply (erasure fires) BUT paraphrase VG-ness")
    print("  stays high (close to baseline) → style bypasses the erasure.")
    print("ROBUST (honest negative):")
    print("  Paraphrase VG-ness drops as much as named → SPEED resists paraphrasing.")
    print("Compare SPEED vs ESD to see if the null-space adds robustness or not.")

    # --- Compute and display tier averages ---
    print("\n=== TIER AVERAGES ===")
    tiers = {}
    for pid, cat, role, vgn, dft, ns in rows:
        tier = cat
        if tier not in tiers:
            tiers[tier] = {m: [] for m in ["baseline"] + METHODS}
        for m in ["baseline"] + METHODS:
            v = vgn.get(m, float("nan"))
            if v == v:  # not nan
                tiers[tier][m].append(v)

    for tier in ["named", "paraphrase_desc", "paraphrase_style", "paraphrase_abstract", "control_far"]:
        if tier not in tiers:
            continue
        line = f"  {tier:<24}"
        for m in ["baseline"] + METHODS:
            vals = tiers[tier][m]
            avg = np.mean(vals) if vals else float("nan")
            line += f"  {m}={avg:.4f}"
        print(line)


if __name__ == "__main__":
    main()
