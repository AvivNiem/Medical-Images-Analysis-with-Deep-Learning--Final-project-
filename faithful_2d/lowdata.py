"""
faithful_2d/lowdata.py
---------------------
Low-data-regime sweep on the faithful 2D spleen pipeline (deep supervision + grid
attention + affine aug + z-score). Directly tests the original paper's claim that
attention helps most when training data is scarce.

Design: fix a held-out TEST set and a small VAL set (for early stopping); from the
remaining training pool, train each variant at several training-set SIZES (4, 8,
16, all patients) across multiple seeds, and evaluate on the same fixed test set.
The subset at each size is seeded so all variants see the SAME patients at that
size (fair comparison). Produces a Dice-vs-training-size curve per variant.

    python -m faithful_2d.lowdata --slice-dir ./slices_2d_256 \
           --sizes 4 8 16 all --seeds 0 1 2 --epochs 60
"""
import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.losses import build_loss
from .utils import set_seed, get_device
from .model import build_faithful_model, count_parameters
from .data import list_patients
from .train import FaithfulKFoldConfig, _loaders, _val_dice, _evaluate, _dsv_loss

NAMES = {"None": "U-Net", "spatial": "Attention U-Net", "cbam": "AG+CBAM", "hybrid": "Hybrid(ours)"}
ORDER = [None, "spatial", "cbam", "hybrid"]


def train_one(att, cfg, train_ids, val_ids, test_ids, device, seed):
    """Train a single (variant, subset, seed) and return test metrics."""
    set_seed(seed)
    train_loader, val_loader, test_loader = _loaders(cfg, train_ids, val_ids, test_ids, seed)
    model = build_faithful_model(att, base_channels=cfg.base_channels,
                                 deep_supervision=cfg.deep_supervision).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = build_loss(cfg.loss_name)

    best, best_state, no_improve = -1.0, None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        for img, msk in train_loader:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad()
            main, aux = model(img)
            loss = _dsv_loss(main, aux, msk, criterion, cfg.dsv_weight)
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
    if best_state is not None:
        model.load_state_dict(best_state)
    return _evaluate(model, test_loader, device)


def run_sweep(cfg, train_sizes, seeds, out_dir):
    device = get_device()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    patients = list_patients(cfg.slice_dir)

    # Fixed test + val; the rest is the training pool.
    idx = np.arange(len(patients)); np.random.RandomState(cfg.split_seed).shuffle(idx)
    n_test = max(1, int(0.2 * len(patients)))
    n_val = max(1, int(0.15 * len(patients)))
    test_ids = [patients[i] for i in idx[:n_test]]
    val_ids = [patients[i] for i in idx[n_test:n_test + n_val]]
    pool = [patients[i] for i in idx[n_test + n_val:]]
    print(f"{len(patients)} patients -> test {len(test_ids)}, val {len(val_ids)}, "
          f"train pool {len(pool)}")

    runs = []
    for ts in train_sizes:
        for att in cfg.attention_types:
            for seed in seeds:
                sub_rng = random.Random(1000 + seed)
                train_ids = pool if ts is None else sub_rng.sample(pool, min(ts, len(pool)))
                m = train_one(att, cfg, train_ids, val_ids, test_ids, device, seed)
                label = "all" if ts is None else ts
                runs.append({"attention_type": str(att), "train_size": label, "seed": seed,
                             "dice": m["dice_mean"], "precision": m["precision_mean"],
                             "recall": m["recall_mean"]})
                print(f"  size={label:>3} {str(att):8s} seed={seed} -> Dice {m['dice_mean']:.4f}")
                json.dump({"config": asdict(cfg), "runs": runs},
                          open(out_dir / "lowdata_results.json", "w"), indent=2)

    # Aggregate + plot Dice vs training size per variant.
    sizes = [("all" if s is None else s) for s in train_sizes]
    numeric = [s for s in sizes if s != "all"]
    step = (numeric[-1] - numeric[-2]) if len(numeric) > 1 else (numeric[-1] if numeric else 1)
    xpos = {s: (numeric[-1] + step if s == "all" else s) for s in sizes}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    print("\n===== Low-data sweep summary (mean Dice over seeds) =====")
    for att in ORDER:
        a = str(att)
        xs, ys, es = [], [], []
        for s in sizes:
            vals = [r["dice"] for r in runs if r["attention_type"] == a and r["train_size"] == s]
            if vals:
                xs.append(xpos[s]); ys.append(np.mean(vals)); es.append(np.std(vals))
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=NAMES[a])
        print(f"  {NAMES[a]:16s} " + "  ".join(
            f"n={s}:{np.mean([r['dice'] for r in runs if r['attention_type']==a and r['train_size']==s]):.3f}"
            for s in sizes))
    ax.set_xlabel("Number of training patients"); ax.set_ylabel("Test Dice")
    ax.set_title("Low-data regime — faithful 2D spleen (mean ± std over seeds)")
    ax.set_xticks([xpos[s] for s in sizes]); ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "fig_lowdata_faithful.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved lowdata_results.json and fig_lowdata_faithful.png to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", default="./slices_2d_256")
    ap.add_argument("--sizes", nargs="+", default=["4", "8", "16", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out-dir", default="./results_lowdata")
    args = ap.parse_args()
    sizes = [None if s == "all" else int(s) for s in args.sizes]
    cfg = FaithfulKFoldConfig(slice_dir=args.slice_dir, max_epochs=args.epochs)
    run_sweep(cfg, sizes, args.seeds, args.out_dir)
