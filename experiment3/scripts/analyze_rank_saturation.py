"""
analyze_rank_saturation.py — Exp 3.4 verdict.

Measures CLIP image-image drift = 1 - cos(emb_baseline, emb_speed_Nc) for each
held-out prompt, across N = 5,10,20,40 concentrated impressionist erasures. Black
frames are excluded so they cannot fake a collapse (the Exp 3.3 failure mode); each
cell reports n = valid seed pairs used.

Read it:
  COLLAPSE / rank saturation (a real, novel-for-SPEED limitation) if the canaries
    (gauguin/seurat/pissarro) AND the supertype prompts drift UP monotonically with N
    and their N=40 drift clearly exceeds the style-far controls (rembrandt/hokusai).
  ROBUST (honest negative) if canary+supertype drift stays flat / no larger than
    controls even at N=40 — SPEED's null-space resists concentrated mass erasure.
The erased_check row (renoir) should show high drift at all N (sanity that erasure
actually fires).
"""
import os
import argparse
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (id, category) — must match probe_rank_saturation.py
ARTISTS = [
    ("renoir",        "erased_check",  "erased at every N (sanity)"),
    ("gauguin",       "canary",        "CANARY (retained, impressionist)"),
    ("seurat",        "canary",        "CANARY (retained, impressionist)"),
    ("pissarro",      "canary",        "CANARY (retained, impressionist)"),
    ("impressionist", "supertype",     "SUPERTYPE capability"),
    ("postimpr",      "supertype",     "SUPERTYPE capability"),
    ("pointillist",   "supertype",     "SUPERTYPE capability"),
    ("rembrandt",     "control_far",   "control (style-far)"),
    ("hokusai",       "control_far",   "control (style-far)"),
]
METHODS = ["speed_5c", "speed_10c", "speed_20c", "speed_40c"]


def is_black(path, thresh=10.0):
    px = list(Image.open(path).convert("L").getdata())
    return (sum(px) / len(px)) < thresh


@torch.no_grad()
def embed(model, proc, path):
    e = model.get_image_features(**proc(images=Image.open(path).convert("RGB"),
                                        return_tensors="pt").to(DEVICE))
    return e / e.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiment3/results/rank_saturation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_csv", default="experiment3/results/rank_saturation/rank_drift.csv")
    args = ap.parse_args()

    print(f"Loading CLIP: {args.clip}")
    model = CLIPModel.from_pretrained(args.clip).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(args.clip)

    rows, n_excluded = [], 0
    header = f"{'id':<14}{'role':<34}" + "".join(f"{m:>11}" for m in METHODS) + "   n  rising?"
    print("\n" + header)
    print("-" * len(header))

    for art, cat, role in ARTISTS:
        drift, nvalid = {}, {}
        for method in METHODS:
            ds = []
            for s in args.seeds:
                bp = f"{args.root}/baseline/{cat}/{art}/seed{s}.png"
                ep = f"{args.root}/{method}/{cat}/{art}/seed{s}.png"
                if not (os.path.exists(bp) and os.path.exists(ep)):
                    continue
                if is_black(bp) or is_black(ep):
                    n_excluded += 1
                    continue
                ds.append(1.0 - (embed(model, proc, bp) * embed(model, proc, ep)).sum().item())
            drift[method] = sum(ds) / len(ds) if ds else float("nan")
            nvalid[method] = len(ds)

        prog = [drift[m] for m in METHODS]
        rising = (not any(p != p for p in prog)) and all(prog[i] <= prog[i+1] + 1e-4 for i in range(len(prog)-1))
        nmin = min(nvalid.values())
        cells = "".join(f"{drift[m]:>11.4f}" for m in METHODS)
        print(f"{art:<14}{role:<34}{cells}{nmin:>4}  {'YES' if rising else 'no'}")
        rows.append((art, cat, role, drift, nvalid, rising))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("id,category,role," + ",".join(METHODS) + "," + ",".join(f"n_{m}" for m in METHODS) + ",rising\n")
        for art, cat, role, d, n, r in rows:
            f.write(f"{art},{cat},{role}," + ",".join(f"{d[m]:.5f}" for m in METHODS)
                    + "," + ",".join(str(n[m]) for m in METHODS) + f",{r}\n")
    print(f"\nWrote {args.out_csv}")
    if n_excluded:
        print(f"NOTE: excluded {n_excluded} seed-pair(s) with a corrupt/black frame (should be 0).")

    print("\n=== VERDICT GUIDE ===")
    print("COLLAPSE (rank saturation, real limitation): canaries + supertype drift rise")
    print("  monotonically with N and their N=40 value clearly exceeds rembrandt/hokusai.")
    print("ROBUST (honest negative): canary/supertype drift stays flat or <= controls at N=40.")
    print("Trust a row only where n is close to the number of seeds run.")


if __name__ == "__main__":
    main()
