"""
faithful_2d/visualize_grid.py
----------------------------
Qualitative prediction grid for the faithful 2D models: random test slices (rows)
x the four variants (columns), each panel showing the predicted spleen (red) over
the ground-truth contour (blue), with per-slice Dice. Produced for BOTH the
full-data and low-data (n=8) settings, on the SAME fixed test slices, so the two
figures are directly comparable.

    # full-data models (trained on the whole training pool):
    python -m faithful_2d.visualize_grid --setting full  --slice-dir ./slices_2d_256
    # low-data models (trained on 8 patients):
    python -m faithful_2d.visualize_grid --setting lowdata --slice-dir ./slices_2d_256

Writes results_grids/fig_grid_full.png and fig_grid_lowdata.png.
"""
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.losses import build_loss
from src.metrics import dice_score
from src.train import set_seed, get_device
from .model import build_faithful_model
from .data import FaithfulSpleenDataset, list_patients
from .train import FaithfulKFoldConfig, _loaders, _val_dice, _dsv_loss

VARIANTS = [None, "spatial", "cbam", "hybrid"]
NAMES = {"None": "U-Net", "spatial": "Attention U-Net", "cbam": "AG + CBAM",
         "hybrid": "AG + context-gated (ours)"}


def train_model(att, cfg, train_ids, val_ids, test_ids, device, seed=0):
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
            _dsv_loss(main, aux, msk, crit, cfg.dsv_weight).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        vd = _val_dice(model, val_loader, device)
        if vd > best:
            best, best_state, no_improve = vd, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["full", "lowdata"], required=True)
    ap.add_argument("--slice-dir", default="./slices_2d_256")
    ap.add_argument("--size", type=int, default=8, help="n patients for lowdata setting")
    ap.add_argument("--n-slices", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out-dir", default="./results_grids")
    args = ap.parse_args()

    device = get_device()
    cfg = FaithfulKFoldConfig(slice_dir=args.slice_dir, max_epochs=args.epochs)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Same fixed split as the low-data experiments (so both settings share the test set).
    patients = list_patients(cfg.slice_dir)
    idx = np.arange(len(patients)); np.random.RandomState(cfg.split_seed).shuffle(idx)
    n_test = max(1, int(0.2 * len(patients))); n_val = max(1, int(0.15 * len(patients)))
    test_ids = [patients[i] for i in idx[:n_test]]
    val_ids = [patients[i] for i in idx[n_test:n_test + n_val]]
    pool = [patients[i] for i in idx[n_test + n_val:]]

    if args.setting == "full":
        train_ids = pool
    else:
        train_ids = random.Random(1000).sample(pool, min(args.size, len(pool)))
    print(f"[{args.setting}] training on {len(train_ids)} patients; test {len(test_ids)}")

    # Same test slices for both settings (deterministic pick spread across the range).
    test_ds = FaithfulSpleenDataset(cfg.slice_dir, test_ids, augment=False, zscore=True)
    fg = [i for i in range(len(test_ds)) if test_ds[i][1].sum() > 0]
    picks = [fg[int(k)] for k in np.linspace(len(fg) * 0.1, len(fg) * 0.9, args.n_slices)]

    models = {str(a): train_model(a, cfg, train_ids, val_ids, test_ids, device) for a in VARIANTS}

    ncol = len(VARIANTS) + 1
    fig, axes = plt.subplots(args.n_slices, ncol, figsize=(2.6 * ncol, 2.6 * args.n_slices))
    for r, idx_s in enumerate(picks):
        img, msk = test_ds[idx_s]
        img_np, msk_np = img.squeeze().numpy(), msk.squeeze().numpy()
        ax = axes[r, 0]
        ax.imshow(img_np, cmap="gray"); ax.contour(msk_np, [0.5], colors="deepskyblue", linewidths=1)
        ax.set_ylabel(f"slice {idx_s}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title("Input + GT", fontsize=10)
        for c, a in enumerate(VARIANTS, start=1):
            with torch.no_grad():
                logits = models[str(a)](img.unsqueeze(0).to(device))[0]
                d = dice_score(logits, msk.unsqueeze(0).to(device)).item()
                pred = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
            ax = axes[r, c]
            ax.imshow(img_np, cmap="gray")
            ax.imshow(np.ma.masked_where(pred < 0.5, pred), cmap="autumn", alpha=0.5)
            ax.contour(msk_np, [0.5], colors="deepskyblue", linewidths=1)
            ax.set_title(f"{NAMES[str(a)]}\nDice={d:.2f}" if r == 0 else f"Dice={d:.2f}", fontsize=8)
            ax.axis("off")
    title = ("Full-data models" if args.setting == "full" else f"Low-data models (n={args.size} patients)")
    fig.suptitle(f"{title} — predictions (red) vs. ground truth (blue)", y=1.005)
    fig.tight_layout()
    path = out / f"fig_grid_{args.setting}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
