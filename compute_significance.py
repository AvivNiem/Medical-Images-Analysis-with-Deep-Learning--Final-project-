"""
compute_significance.py
-----------------------
Computes the full pairwise statistical-significance matrix between all four model
variants on the test set, using the weights already saved by run_all.py (no
retraining). Pools per-slice Dice across ALL trained seeds for a more robust
paired Wilcoxon signed-rank test than a single-seed comparison.

Run on the server after run_all.py has finished:
    CUDA_VISIBLE_DEVICES=1 python compute_significance.py --image-size 256
"""

import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from scipy.stats import wilcoxon

from src.data import patient_level_split, SpleenSliceDataset
from src.train import TrainConfig, get_device, load_trained_model
from src.metrics import dice_score, precision_recall

VARIANTS = [None, "spatial", "cbam", "hybrid"]
NAMES = {"None": "U-Net", "spatial": "Attention U-Net", "cbam": "AG+CBAM", "hybrid": "Hybrid(ours)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--results", default="./results/results.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    results = json.load(open(args.results))
    seeds = results["config"]["seeds"]

    slice_dir = Path(f"./slices_2d_{args.image_size}")
    cfg = TrainConfig(slice_dir=str(slice_dir), base_channels=32, image_size=args.image_size)

    _, _, test_ids = patient_level_split(slice_dir, val_frac=cfg.val_frac,
                                         test_frac=cfg.test_frac, seed=1234)
    test_ds = SpleenSliceDataset(slice_dir, test_ids, augment=False)
    fg_idx = [i for i in range(len(test_ds)) if test_ds[i][1].sum() > 0]
    print(f"{len(fg_idx)} foreground test slices x {len(seeds)} seeds "
          f"= {len(fg_idx) * len(seeds)} paired samples per comparison")

    # Per-slice Dice / precision / recall pooled across all seeds, aligned by
    # (seed, slice) so the comparison between any two variants is properly paired.
    per_slice = {str(a): [] for a in VARIANTS}
    prec = {str(a): [] for a in VARIANTS}
    rec = {str(a): [] for a in VARIANTS}
    for seed in seeds:
        for att in VARIANTS:
            model = load_trained_model(results, att, seed, cfg, device)
            with torch.no_grad():
                for i in fg_idx:
                    im, mk = test_ds[i]
                    logits = model(im.unsqueeze(0).to(device))
                    tgt = mk.unsqueeze(0).to(device)
                    per_slice[str(att)].append(dice_score(logits, tgt).item())
                    p, r = precision_recall(logits, tgt)
                    prec[str(att)].append(p.item())
                    rec[str(att)].append(r.item())

    # Mean per variant (Dice / precision / recall), all from the same weights
    print("\nPooled per-slice metrics (mean over all seeds x foreground slices):")
    print(f"  {'Variant':16s} {'Dice':>8s} {'Prec':>8s} {'Rec':>8s}")
    for a in VARIANTS:
        s = str(a)
        print(f"  {NAMES[s]:16s} {np.mean(per_slice[s]):8.4f} {np.mean(prec[s]):8.4f} {np.mean(rec[s]):8.4f}")

    # Pairwise paired Wilcoxon
    print("\nPairwise paired Wilcoxon signed-rank (median diff A-B, p-value):")
    for a, b in combinations([str(x) for x in VARIANTS], 2):
        A, B = np.array(per_slice[a]), np.array(per_slice[b])
        diff = A - B
        nz = diff != 0
        if nz.sum() == 0:
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(A[nz], B[nz])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {NAMES[a]:16s} vs {NAMES[b]:16s}  median diff {np.median(diff):+.4f}  p={p:.4f}  {sig}")


if __name__ == "__main__":
    main()
