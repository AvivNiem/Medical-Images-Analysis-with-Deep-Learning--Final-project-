"""
faithful_2d/train.py
--------------------
Patient-level k-fold cross-validation for the faithful 2D reproduction (deep
supervision + grid attention + affine augmentation + z-score). Reuses the proven
2D loss and metrics from `src/`.

  * Loss: Dice + Focal (kept from the working 2D pipeline — on small-foreground 2D
    slices pure Dice is unstable; documented as a necessary 2D adaptation) with
    deep-supervision auxiliary terms (coarser heads down-weighted).
  * For each of k folds: that fold = TEST, rest = train (with a small val holdout
    for early stopping). Report mean +/- std of test metrics across folds.

    python -m faithful_2d.train --slice-dir ./slices_2d_256 --k 5 --epochs 60
"""
import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.losses import build_loss
from src.metrics import MetricAccumulator, dice_score
from .utils import set_seed, get_device
from .model import build_faithful_model, count_parameters
from .data import FaithfulSpleenDataset, list_patients


@dataclass
class FaithfulKFoldConfig:
    slice_dir: str = "./slices_2d_256"
    attention_types: List[Optional[str]] = field(
        default_factory=lambda: [None, "spatial", "cbam", "hybrid"])
    k: int = 5
    base_channels: int = 32
    batch_size: int = 8
    max_epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    loss_name: str = "dice_focal"
    dsv_weight: float = 0.5
    deep_supervision: bool = True
    patience: int = 12
    num_workers: int = 2
    val_frac_within_train: float = 0.15
    out_dir: str = "./results_faithful2d"
    split_seed: int = 1234


def _dsv_loss(main, aux, target, loss_fn, dsv_w):
    loss = loss_fn(main, target)
    n = len(aux)
    for i, a in enumerate(aux):
        loss = loss + dsv_w * (0.5 ** (n - 1 - i)) * loss_fn(a, target)
    return loss


def _loaders(cfg, train_ids, val_ids, test_ids, seed):
    tr = FaithfulSpleenDataset(cfg.slice_dir, train_ids, augment=True, zscore=True)
    va = FaithfulSpleenDataset(cfg.slice_dir, val_ids, augment=False, zscore=True)
    te = FaithfulSpleenDataset(cfg.slice_dir, test_ids, augment=False, zscore=True)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                       generator=g, drop_last=True),
            DataLoader(va, batch_size=cfg.batch_size, num_workers=cfg.num_workers),
            DataLoader(te, batch_size=cfg.batch_size, num_workers=cfg.num_workers))


@torch.no_grad()
def _val_dice(model, loader, device):
    model.eval(); scores = []
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        has = (msk.reshape(msk.size(0), -1).sum(1) > 0)
        if has.any():
            d = dice_score(model(img)[0], msk)     # main output
            scores.extend(d[has].cpu().numpy().tolist())
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval(); acc = MetricAccumulator(exclude_empty_gt=True)
    for img, msk in loader:
        acc.update(model(img.to(device))[0], msk.to(device))   # main output
    return acc.summary()


def train_fold(att, cfg, fold, train_ids, val_ids, test_ids, device, weights_dir):
    set_seed(fold)
    train_loader, val_loader, test_loader = _loaders(cfg, train_ids, val_ids, test_ids, fold)
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
            if cfg.grad_clip:
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
    metrics = _evaluate(model, test_loader, device)
    ckpt = weights_dir / f"{att}_fold{fold}.pt"
    torch.save(best_state, ckpt)
    print(f"  [{str(att):8s} fold{fold}] val_best={best:.4f} test Dice {metrics['dice_mean']:.4f} "
          f"prec {metrics['precision_mean']:.4f} rec {metrics['recall_mean']:.4f}")
    return {"attention_type": str(att), "fold": fold, "test_metrics": metrics,
            "num_params": count_parameters(model), "weights_path": str(ckpt), "test_ids": test_ids}


def run_faithful_kfold(cfg: FaithfulKFoldConfig):
    device = get_device()
    print(f"Device: {device}")
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"; weights_dir.mkdir(exist_ok=True)
    patients = list_patients(cfg.slice_dir)
    idx = np.arange(len(patients)); np.random.RandomState(cfg.split_seed).shuffle(idx)
    folds = np.array_split(idx, cfg.k)
    print(f"{len(patients)} patients, {cfg.k}-fold CV, variants={[str(a) for a in cfg.attention_types]}")

    runs = []
    for att in cfg.attention_types:
        for f in range(cfg.k):
            test_ids = [patients[i] for i in folds[f]]
            trainval = [patients[i] for j in range(cfg.k) if j != f for i in folds[j]]
            n_val = max(1, int(cfg.val_frac_within_train * len(trainval)))
            val_ids, train_ids = trainval[:n_val], trainval[n_val:]
            runs.append(train_fold(att, cfg, f, train_ids, val_ids, test_ids, device, weights_dir))
            json.dump({"config": asdict(cfg), "runs": runs},
                      open(out_dir / "results_faithful2d.json", "w"), indent=2)

    print("\n===== Faithful 2D k-fold summary (mean ± std over folds) =====")
    for att in cfg.attention_types:
        ds = [r["test_metrics"]["dice_mean"] for r in runs if r["attention_type"] == str(att)]
        print(f"  {str(att):8s} Dice {np.mean(ds):.4f} ± {np.std(ds):.4f}")
    print(f"\nSaved to {out_dir/'results_faithful2d.json'}")
    return {"config": asdict(cfg), "runs": runs}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", default="./slices_2d_256")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out-dir", default="./results_faithful2d")
    ap.add_argument("--debug1", action="store_true", help="train None fold0 only, to check convergence")
    args = ap.parse_args()
    cfg = FaithfulKFoldConfig(slice_dir=args.slice_dir, k=args.k, max_epochs=args.epochs, out_dir=args.out_dir)
    if args.debug1:
        import torch as _t
        dev = get_device()
        # shape check
        m = build_faithful_model(None, base_channels=cfg.base_channels)
        o, a = m(_t.randn(2, 1, 256, 256))
        print(f"shape check: main={tuple(o.shape)} aux={len(a)} params={count_parameters(m):,}")
        pats = list_patients(cfg.slice_dir)
        idx = np.arange(len(pats)); np.random.RandomState(cfg.split_seed).shuffle(idx)
        folds = np.array_split(idx, cfg.k)
        test_ids = [pats[i] for i in folds[0]]
        trainval = [pats[i] for j in range(1, cfg.k) for i in folds[j]]
        nv = max(1, int(0.15 * len(trainval)))
        wd = Path(cfg.out_dir) / "weights"; wd.mkdir(parents=True, exist_ok=True)
        r = train_fold(None, cfg, 0, trainval[nv:], trainval[:nv], test_ids, dev, wd)
        print(f"DEBUG1 faithful-2D: None fold0 test Dice {r['test_metrics']['dice_mean']:.4f}")
    else:
        run_faithful_kfold(cfg)
