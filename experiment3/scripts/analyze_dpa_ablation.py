"""
analyze_dpa_ablation.py — Experiment 3.5 verdict.

At N=40, compare CLIP image-image drift (vs baseline) of the held-out canaries and
controls across three SPEED refinement settings:
    speed_40c          full method (IPF + DPA)
    speed_40c_nodpa    IPF on, DPA off  (DPA isolated via --disable_dpa)
    speed_40c_norefine neither IPF nor DPA (aug_num 0)

The key row is PISSARRO (the explicit retain-set member that leaks under the full
method). Reading:
    drift(nodpa)  < drift(full)  -> DPA ACCELERATES the collapse (its augmentation
                                    inflates retain rank, shrinking the null space).
    drift(nodpa)  > drift(full)  -> DPA HELPS even under load (robustness for SPEED).
    drift(nodpa) == drift(full)  -> DPA's benefit doesn't reach the entangled member.

Black frames are excluded; n = valid seed pairs per cell.
"""
import os
import argparse
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARTISTS = [
    ("renoir",        "erased_check",  "erased at N=40 (sanity)"),
    ("pissarro",      "canary",        "CANARY — the leaker (retain member)"),
    ("gauguin",       "canary",        "canary — post-impressionist"),
    ("seurat",        "canary",        "canary — pointillist"),
    ("impressionist", "supertype",     "supertype capability"),
    ("rembrandt",     "control_far",   "control — style-far"),
    ("hokusai",       "control_far",   "control — style-far"),
]
METHODS = ["speed_40c", "speed_40c_nodpa", "speed_40c_norefine"]


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
    ap.add_argument("--out_csv", default="experiment3/results/rank_saturation/dpa_ablation.csv")
    args = ap.parse_args()

    print(f"Loading CLIP: {args.clip}")
    model = CLIPModel.from_pretrained(args.clip).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(args.clip)

    rows, n_excluded = [], 0
    hdr = f"{'id':<14}{'role':<34}{'full':>12}{'no-DPA':>12}{'no-refine':>12}   n"
    print("\n" + hdr); print("-" * len(hdr))

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
        nmin = min(nvalid.values())
        cells = "".join(f"{drift[m]:>12.4f}" for m in METHODS)
        print(f"{art:<14}{role:<34}{cells}{nmin:>4}")
        rows.append((art, cat, role, drift, nvalid))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("id,category,role," + ",".join(METHODS) + "," + ",".join(f"n_{m}" for m in METHODS) + "\n")
        for art, cat, role, d, n in rows:
            f.write(f"{art},{cat},{role}," + ",".join(f"{d[m]:.5f}" for m in METHODS)
                    + "," + ",".join(str(n[m]) for m in METHODS) + "\n")
    print(f"\nWrote {args.out_csv}")
    if n_excluded:
        print(f"NOTE: excluded {n_excluded} seed-pair(s) with a corrupt/black frame (should be 0).")

    print("\n=== VERDICT (read the PISSARRO row) ===")
    print("no-DPA < full  -> DPA accelerates the collapse (coverage component hurts capacity).")
    print("no-DPA > full  -> DPA helps even under load (robustness result for SPEED).")
    print("no-DPA ~ full  -> DPA's benefit doesn't reach the entangled retain member.")


if __name__ == "__main__":
    main()
