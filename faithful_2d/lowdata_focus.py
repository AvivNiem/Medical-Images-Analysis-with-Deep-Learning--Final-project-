"""
faithful_2d/lowdata_focus.py
---------------------------
Focused low-data experiment at a single training size (default n=8 patients) with
MANY seeds and a per-slice paired significance test. Purpose: determine whether
channel attention improves over the PAPER'S model (the spatial Attention U-Net) in
the low-data regime — i.e. is `cbam` (or our `hybrid`) significantly better than
`spatial` at n=8?

For each seed: sample the SAME n-patient training subset for all four variants
(fair), train each, and score per-slice Dice on the FIXED test set. Pool per-slice
Dice across seeds (aligned by seed+slice) and run pairwise Wilcoxon signed-rank.

    python -m faithful_2d.lowdata_focus --slice-dir ./slices_2d_256 \
           --size 8 --n-seeds 8 --epochs 60
"""
import argparse
import copy
import json
from itertools import combinations
from pathlib import Path
import random

import numpy as np
import torch
from scipy.stats import wilcoxon

from src.losses import build_loss
from src.metrics import dice_score, precision_recall
from .utils import set_seed, get_device
from .model import build_faithful_model
from .data import FaithfulSpleenDataset, list_patients
from .train import FaithfulKFoldConfig, _loaders, _val_dice, _dsv_loss

VARIANTS = [None, "spatial", "cbam", "hybrid"]
NAMES = {"None": "U-Net", "spatial": "Attention U-Net", "cbam": "AG+CBAM", "hybrid": "Hybrid(ours)"}


def train_and_perslice(att, cfg, train_ids, val_ids, test_ids, device, seed, fg_idx, test_ds):
    set_seed(seed)
    train_loader, val_loader, _ = _loaders(cfg, train_ids, val_ids, test_ids, seed)
    model = build_faithful_model(att, base_channels=cfg.base_channels,
                                 deep_supervision=cfg.deep_supervision).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = build_loss(cfg.loss_name)
    best, best_state, no_improve = -1.0, None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        for img, msk in train_loader:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad()
            main, aux = model(img)
            loss = _dsv_loss(main, aux, msk, crit, cfg.dsv_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        vd = _val_dice(model, val_loader, device)
        if vd > best:
            best, best_state, no_improve = vd, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    per_d, per_p, per_r = [], [], []
    with torch.no_grad():
        for i in fg_idx:
            img, msk = test_ds[i]
            logits = model(img.unsqueeze(0).to(device))[0]
            tgt = msk.unsqueeze(0).to(device)
            per_d.append(dice_score(logits, tgt).item())
            p, r = precision_recall(logits, tgt)
            per_p.append(p.item()); per_r.append(r.item())
    return per_d, per_p, per_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", default="./slices_2d_256")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out-dir", default="./results_lowdata_focus")
    args = ap.parse_args()

    device = get_device()
    cfg = FaithfulKFoldConfig(slice_dir=args.slice_dir, max_epochs=args.epochs)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    patients = list_patients(cfg.slice_dir)
    idx = np.arange(len(patients)); np.random.RandomState(cfg.split_seed).shuffle(idx)
    n_test = max(1, int(0.2 * len(patients)))
    n_val = max(1, int(0.15 * len(patients)))
    test_ids = [patients[i] for i in idx[:n_test]]
    val_ids = [patients[i] for i in idx[n_test:n_test + n_val]]
    pool = [patients[i] for i in idx[n_test + n_val:]]
    test_ds = FaithfulSpleenDataset(cfg.slice_dir, test_ids, augment=False, zscore=True)
    fg_idx = [i for i in range(len(test_ds)) if test_ds[i][1].sum() > 0]
    print(f"n={args.size} patients, {args.n_seeds} seeds, {len(fg_idx)} fixed test fg slices "
          f"-> {args.n_seeds * len(fg_idx)} paired samples/comparison")

    per_slice = {str(a): [] for a in VARIANTS}
    prec = {str(a): [] for a in VARIANTS}
    rec = {str(a): [] for a in VARIANTS}
    for seed in range(args.n_seeds):
        subset = random.Random(1000 + seed).sample(pool, min(args.size, len(pool)))
        for att in VARIANTS:
            pd, pp, pr = train_and_perslice(att, cfg, subset, val_ids, test_ids, device, seed, fg_idx, test_ds)
            per_slice[str(att)].extend(pd); prec[str(att)].extend(pp); rec[str(att)].extend(pr)
            print(f"  seed{seed} {str(att):8s} Dice {np.mean(pd):.4f}  Prec {np.mean(pp):.4f}  Rec {np.mean(pr):.4f}")
        json.dump({"dice": per_slice, "precision": prec, "recall": rec}, open(out / "perslice.json", "w"))

    print(f"\n===== n={args.size}: mean over {args.n_seeds} seeds x slices =====")
    print(f"  {'Variant':16s} {'Dice':>8s} {'Prec':>8s} {'Rec':>8s}")
    for a in VARIANTS:
        s = str(a)
        print(f"  {NAMES[s]:16s} {np.mean(per_slice[s]):8.4f} {np.mean(prec[s]):8.4f} {np.mean(rec[s]):8.4f}")

    print("\nPairwise paired Wilcoxon (median A-B, p) — focus: vs Attention U-Net (paper):")
    for a, b in combinations([str(x) for x in VARIANTS], 2):
        A, B = np.array(per_slice[a]), np.array(per_slice[b]); nz = (A - B) != 0
        stat, p = (wilcoxon(A[nz], B[nz]) if nz.sum() else (float("nan"), 1.0))
        tag = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {NAMES[a]:16s} vs {NAMES[b]:16s}  {np.median(A - B):+.4f}  p={p:.4f}  {tag}")


if __name__ == "__main__":
    main()
