"""
analyze_clip_drift.py — measure how far each artist's generation drifts from the
baseline as SPEED erases more concepts simultaneously (1c -> 2c -> 3c).

WHY NOT PIXEL MSE: pixel MSE is dominated by seed-to-seed composition noise. In
Experiment 3.2 a *retained* artist (Cezanne) scored pixel-MSE > 10,000 on some
seeds while a supposedly "damaged" artist scored ~440 — i.e. MSE cannot tell
style damage from ordinary diffusion stochasticity.

WHAT WE MEASURE INSTEAD: CLIP image-image cosine drift = 1 - cos(emb_baseline,
emb_edited) for the SAME prompt and SAME seed. CLIP embeds style, so a genuine
style change shows up here even when composition is unchanged.

THE ACTUAL SIGNAL: we do not care about absolute drift magnitude (CLIP has its own
noise floor). We care whether drift increases MONOTONICALLY across 1c -> 2c -> 3c
for the style-adjacent retain canaries (Gauguin/Seurat/Pissarro), while the
style-FAR retain controls (Rembrandt/Hokusai) stay flat. Random degradation is not
monotonic and would hit canaries and controls equally; concentrated null-space
pressure is monotonic and hits only the style-adjacent retain members.
"""
import os
import argparse
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARTISTS = [
    ("vangogh",      "erased",              "erased target (VG, all configs)"),
    ("picasso",      "erased_in_2c_3c",     "erased target (2c, 3c)"),
    ("monet",        "erased_in_3c",        "erased target (3c only)"),
    ("gauguin",      "retain_canary",       "RETAIN canary (style-adjacent)"),
    ("seurat",       "retain_canary",       "RETAIN canary (style-adjacent)"),
    ("pissarro",     "retain_canary",       "RETAIN canary (style-adjacent)"),
    ("rembrandt",    "retain_control_far",  "RETAIN control (style-far)"),
    ("hokusai",      "retain_control_far",  "RETAIN control (style-far)"),
    ("rysselberghe", "non_retain",          "non-retain bridge"),
]
METHODS = ["speed_1c", "speed_2c", "speed_3c", "esd_x"]


@torch.no_grad()
def embed(model, processor, path):
    img = Image.open(path).convert("RGB")
    inp = processor(images=img, return_tensors="pt").to(DEVICE)
    e = model.get_image_features(**inp)
    return e / e.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiment3/results/multi_concept")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_csv", default="experiment3/results/multi_concept/clip_drift.csv")
    args = ap.parse_args()

    print(f"Loading CLIP: {args.clip}")
    model = CLIPModel.from_pretrained(args.clip).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(args.clip)

    rows = []
    header = f"{'artist':<14}{'role':<32}" + "".join(f"{m:>12}" for m in METHODS) + "   monotonic?"
    print("\n" + header)
    print("-" * len(header))

    for art, cat, role in ARTISTS:
        drift_by_method = {}
        for method in METHODS:
            seed_drifts = []
            for s in args.seeds:
                bp = f"{args.root}/baseline/{cat}/{art}/seed{s}.png"
                ep = f"{args.root}/{method}/{cat}/{art}/seed{s}.png"
                if os.path.exists(bp) and os.path.exists(ep):
                    cos = (embed(model, processor, bp) * embed(model, processor, ep)).sum().item()
                    seed_drifts.append(1.0 - cos)
            drift_by_method[method] = sum(seed_drifts) / len(seed_drifts) if seed_drifts else float("nan")

        # Monotonicity check across the SPEED progression only (1c <= 2c <= 3c)
        prog = [drift_by_method[m] for m in ("speed_1c", "speed_2c", "speed_3c")]
        mono = all(prog[i] <= prog[i + 1] + 1e-4 for i in range(len(prog) - 1)) and not any(p != p for p in prog)
        flag = "  YES (rising)" if mono else "  no"

        cells = "".join(f"{drift_by_method[m]:>12.4f}" for m in METHODS)
        print(f"{art:<14}{role:<32}{cells}{flag}")
        rows.append((art, cat, role, drift_by_method, mono))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("artist,category,role," + ",".join(METHODS) + ",speed_monotonic_rising\n")
        for art, cat, role, d, mono in rows:
            f.write(f"{art},{cat},{role}," + ",".join(f"{d[m]:.5f}" for m in METHODS) + f",{mono}\n")
    print(f"\nWrote {args.out_csv}")

    print("\n=== HOW TO READ THIS ===")
    print("FINDING (null-space collapse) if: gauguin/seurat/pissarro show monotonic")
    print("  rising drift AND their speed_3c drift clearly exceeds rembrandt/hokusai's.")
    print("NEGATIVE RESULT (SPEED robust) if: canary drift is flat across 1c/2c/3c, or")
    print("  no larger than the style-far controls. Either way it's an honest result.")


if __name__ == "__main__":
    main()
