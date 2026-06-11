"""
analyze_fidelity.py — Experiment 4: does SPEED preserve a neighbor's IDENTITY but
degrade its FIDELITY?

Motivation: SPEED's preservation is judged (here and in its paper) by metrics that
read SEMANTIC identity — CLIP similarity / classifier scores. But a retained concept
can read as "still itself" to CLIP while its image quality degrades (artifacts, blur,
texture/structure breakage). CLIP cosine is largely blind to this; the erasure
benchmarks EraseBench and EMMA treat image-quality degradation as a SEPARATE failure
axis from semantic preservation. SPEED explicitly claims high fidelity, so this is an
in-scope, untested axis.

What we measure, per (edited model, artist), paired against the baseline at the SAME
seed (so composition is held roughly fixed by the shared initial noise):
  * CLIP image-image drift  = 1 - cos(emb_base, emb_edit)   -> SEMANTIC change (low = identity preserved)
  * LPIPS(base, edit)                                        -> PERCEPTUAL/fidelity change (the fidelity metric)

The finding lives in the CONTRAST, not the absolute LPIPS:
  FIDELITY DEGRADATION (a distinct limitation) iff the retained canaries
    (gauguin/seurat/pissarro) keep LOW CLIP drift (identity preserved, ~ controls)
    but show LPIPS clearly ABOVE the style-far controls (rembrandt/hokusai) — i.e.
    perceptual damage that the semantic metric cannot see.
  ROBUST (honest negative / bounding result) iff canary LPIPS is no higher than the
    controls' — SPEED preserves identity AND fidelity; the CLIP-robustness already
    shown extends to perceptual quality.

Runs purely on already-generated images (experiment3/results/multi_concept) — no
diffusion generation, no new checkpoints. Black/corrupt frames are excluded.
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (id, category) — present in experiment3/results/multi_concept
ARTISTS = [
    ("gauguin",   "retain_canary",      "CANARY — retained, impressionist"),
    ("seurat",    "retain_canary",      "CANARY — retained, pointillist"),
    ("pissarro",  "retain_canary",      "CANARY — retained, core impressionist"),
    ("rembrandt", "retain_control_far", "control — style-far"),
    ("hokusai",   "retain_control_far", "control — style-far"),
]
METHODS = ["speed_1c", "speed_2c", "speed_3c", "esd_x"]


def is_black(path, thresh=10.0):
    px = list(Image.open(path).convert("L").getdata())
    return (sum(px) / len(px)) < thresh


@torch.no_grad()
def clip_emb(model, proc, path):
    e = model.get_image_features(**proc(images=Image.open(path).convert("RGB"),
                                        return_tensors="pt").to(DEVICE))
    return e / e.norm(dim=-1, keepdim=True)


def to_lpips_tensor(path):
    a = np.asarray(Image.open(path).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0  # [-1,1], [1,3,256,256]
    return t.to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiment3/results/multi_concept")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--out_csv", default="experiment3/results/fidelity/fidelity.csv")
    args = ap.parse_args()

    print(f"Loading CLIP: {args.clip}")
    clip_model = CLIPModel.from_pretrained(args.clip).to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip)
    import lpips
    print(f"Loading LPIPS: {args.lpips_net}")
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(DEVICE).eval()

    rows, n_excluded = [], 0
    hdr = (f"{'artist':<12}{'role':<34}"
           + "".join(f"{m+' CLIP':>13}{m+' LPIPS':>13}" for m in METHODS))
    print("\n" + hdr)
    print("-" * len(hdr))

    for art, cat, role in ARTISTS:
        cell = {}
        for method in METHODS:
            clip_ds, lpips_ds = [], []
            for s in args.seeds:
                bp = f"{args.root}/baseline/{cat}/{art}/seed{s}.png"
                ep = f"{args.root}/{method}/{cat}/{art}/seed{s}.png"
                if not (os.path.exists(bp) and os.path.exists(ep)):
                    continue
                if is_black(bp) or is_black(ep):
                    n_excluded += 1
                    continue
                clip_ds.append(1.0 - (clip_emb(clip_model, clip_proc, bp)
                                      * clip_emb(clip_model, clip_proc, ep)).sum().item())
                with torch.no_grad():
                    lpips_ds.append(lpips_model(to_lpips_tensor(bp), to_lpips_tensor(ep)).item())
            cell[method] = (
                sum(clip_ds) / len(clip_ds) if clip_ds else float("nan"),
                sum(lpips_ds) / len(lpips_ds) if lpips_ds else float("nan"),
                len(clip_ds),
            )
        line = f"{art:<12}{role:<34}"
        for m in METHODS:
            c, l, _ = cell[m]
            line += f"{c:>13.4f}{l:>13.4f}"
        print(line)
        rows.append((art, cat, role, cell))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        cols = []
        for m in METHODS:
            cols += [f"{m}_clip", f"{m}_lpips", f"n_{m}"]
        f.write("artist,category,role," + ",".join(cols) + "\n")
        for art, cat, role, cell in rows:
            vals = []
            for m in METHODS:
                c, l, n = cell[m]
                vals += [f"{c:.5f}", f"{l:.5f}", str(n)]
            f.write(f"{art},{cat},{role}," + ",".join(vals) + "\n")
    print(f"\nWrote {args.out_csv}")
    if n_excluded:
        print(f"NOTE: excluded {n_excluded} seed-pair(s) with a corrupt/black frame.")

    print("\n=== VERDICT GUIDE (compare SPEED canaries vs style-far controls) ===")
    print("FIDELITY DEGRADATION (limitation): canary CLIP drift ~ controls (identity preserved)")
    print("  BUT canary LPIPS clearly ABOVE controls -> perceptual damage CLIP cannot see.")
    print("ROBUST (honest negative): canary LPIPS no higher than controls -> fidelity preserved too.")
    print("ESD column is the no-retain-mechanism reference (expected high on both metrics).")
    print("Caveat: same-seed pairing holds composition roughly fixed, but LPIPS still")
    print("  captures some layout change; rely on the canary-vs-control CONTRAST, not absolute LPIPS.")


if __name__ == "__main__":
    main()
