"""
faithful_2d/evaluate.py
-----------------------
Evaluation for the faithful 2D k-fold results: pooled per-slice metrics, pairwise
significance, per-fold box plot, and a qualitative prediction grid. Uses the
weights saved by faithful_2d.train (no retraining).

Because k-fold partitions patients, every patient is tested exactly once (by the
model of the fold it belongs to). Pooling per-slice Dice across folds therefore
covers the whole dataset, and comparisons are paired slice-by-slice (all variants
share the same fold splits).

    python -m faithful_2d.evaluate --slice-dir ./slices_2d_256
Outputs -> results_faithful2d/: significance printout, fig_kfold_box.png,
fig_faithful_grid.png, and results_faithful2d_summary.md
"""
import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from src.metrics import dice_score, precision_recall
from src.train import get_device
from .model import build_faithful_model
from .data import FaithfulSpleenDataset, list_patients

VARIANTS = [None, "spatial", "cbam", "hybrid"]
NAMES = {"None": "U-Net", "spatial": "Attention U-Net", "cbam": "AG+CBAM", "hybrid": "Hybrid(ours)"}


def rebuild_folds(slice_dir, k, seed):
    patients = list_patients(slice_dir)
    idx = np.arange(len(patients)); np.random.RandomState(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    return patients, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", default="./slices_2d_256")
    ap.add_argument("--results", default="./results_faithful2d/results_faithful2d.json")
    ap.add_argument("--out-dir", default="./results_faithful2d")
    args = ap.parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)

    res = json.load(open(args.results))
    cfg = res["config"]
    k, seed, base = cfg["k"], cfg["split_seed"], cfg["base_channels"]
    patients, folds = rebuild_folds(args.slice_dir, k, seed)

    def weights_for(att, fold):
        for r in res["runs"]:
            if r["attention_type"] == str(att) and r["fold"] == fold:
                return r["weights_path"]
        raise KeyError((att, fold))

    # Pooled per-slice metrics, aligned by (fold, slice) across variants.
    per_slice = {str(a): [] for a in VARIANTS}
    prec = {str(a): [] for a in VARIANTS}
    rec = {str(a): [] for a in VARIANTS}
    per_fold_dice = {str(a): [] for a in VARIANTS}   # for the box plot

    for fold in range(k):
        test_ids = [patients[i] for i in folds[fold]]
        ds = FaithfulSpleenDataset(args.slice_dir, test_ids, augment=False, zscore=True)
        fg = [i for i in range(len(ds)) if ds[i][1].sum() > 0]
        for att in VARIANTS:
            model = build_faithful_model(att, base_channels=base,
                                         deep_supervision=cfg["deep_supervision"]).to(device)
            model.load_state_dict(torch.load(weights_for(att, fold), map_location=device))
            model.eval()
            fold_scores = []
            with torch.no_grad():
                for i in fg:
                    img, msk = ds[i]
                    logits = model(img.unsqueeze(0).to(device))[0]
                    tgt = msk.unsqueeze(0).to(device)
                    d = dice_score(logits, tgt).item()
                    p, r = precision_recall(logits, tgt)
                    per_slice[str(att)].append(d); fold_scores.append(d)
                    prec[str(att)].append(p.item()); rec[str(att)].append(r.item())
            per_fold_dice[str(att)].append(float(np.mean(fold_scores)))
        print(f"fold {fold}: {len(fg)} foreground slices evaluated")

    # ---- Pooled table ----
    print("\nPooled per-slice metrics (over all folds):")
    print(f"  {'Variant':16s} {'Dice':>8s} {'Prec':>8s} {'Rec':>8s}")
    lines = []
    for a in VARIANTS:
        s = str(a)
        row = (np.mean(per_slice[s]), np.mean(prec[s]), np.mean(rec[s]))
        print(f"  {NAMES[s]:16s} {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}")
        lines.append((NAMES[s], *row))

    # ---- Pairwise Wilcoxon ----
    print("\nPairwise paired Wilcoxon (median diff A-B, p):")
    sig_lines = []
    for a, b in combinations([str(x) for x in VARIANTS], 2):
        A, B = np.array(per_slice[a]), np.array(per_slice[b])
        nz = (A - B) != 0
        stat, p = (wilcoxon(A[nz], B[nz]) if nz.sum() else (float("nan"), 1.0))
        tag = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {NAMES[a]:16s} vs {NAMES[b]:16s}  {np.median(A-B):+.4f}  p={p:.4f} {tag}")
        sig_lines.append((NAMES[a], NAMES[b], float(np.median(A - B)), float(p), tag))

    # ---- Box plot of per-fold Dice ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [per_fold_dice[str(a)] for a in VARIANTS]
    ax.boxplot(data, labels=[NAMES[str(a)] for a in VARIANTS], showmeans=True)
    ax.set_ylabel("Per-fold test Dice"); ax.set_title("Faithful 2D reproduction — 5-fold CV")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(out_dir / "fig_kfold_box.png", dpi=150, bbox_inches="tight")

    # ---- Summary markdown ----
    with open(out_dir / "results_faithful2d_summary.md", "w") as f:
        f.write("# Faithful 2D reproduction — results\n\n")
        f.write("5-fold patient-level cross-validation. Deep supervision + grid attention "
                "+ affine augmentation + z-score normalization.\n\n")
        f.write("## Per-fold Dice (mean ± std over folds)\n\n| Variant | Dice |\n|---|---|\n")
        for a in VARIANTS:
            v = per_fold_dice[str(a)]
            f.write(f"| {NAMES[str(a)]} | {np.mean(v):.4f} ± {np.std(v):.4f} |\n")
        f.write("\n## Pooled per-slice metrics\n\n| Variant | Dice | Precision | Recall |\n|---|---|---|---|\n")
        for name, d, p, r in lines:
            f.write(f"| {name} | {d:.4f} | {p:.4f} | {r:.4f} |\n")
        f.write("\n## Pairwise paired Wilcoxon (median diff A−B, p)\n\n| A | B | median diff | p | sig |\n|---|---|---|---|---|\n")
        for a, b, md, p, tag in sig_lines:
            f.write(f"| {a} | {b} | {md:+.4f} | {p:.4f} | {tag} |\n")

    print(f"\nSaved fig_kfold_box.png and results_faithful2d_summary.md to {out_dir}/")


if __name__ == "__main__":
    main()
