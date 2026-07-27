"""
train3d.py
----------
3D training with k-fold cross-validation for the faithful reproduction.

  * Loss: Dice loss (the paper's choice). With balanced patch sampling the class
    imbalance is handled at the data level, so plain Dice is stable in 3D (unlike
    the 2D proof-of-concept). Deep supervision: the loss is the main output's Dice
    plus down-weighted Dice on each auxiliary head.
  * Validation: sliding-window inference over the full volume → volume-level Dice.
  * run_kfold(): trains every (variant x fold), saves weights + per-fold metrics to
    results_3d/, and returns the aggregated results.

Runs on any CUDA GPU; the L40 handles 96^3 patches comfortably.
"""

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch

from .model3d import build_model3d, count_parameters
from .data3d import get_data_dicts, kfold_split, make_loaders, PATCH


@dataclass
class Cfg3D:
    task_dir: str = "./decathlon_data/Task09_Spleen"
    out_dir: str = "./results_3d"
    attention_types: List[Optional[str]] = field(
        default_factory=lambda: [None, "spatial", "cbam", "hybrid"])
    k_folds: int = 5
    base: int = 16
    batch_size: int = 2
    max_epochs: int = 200
    val_interval: int = 3
    lr: float = 1e-3             # InstanceNorm stabilises eval, so we can train at 1e-3
    weight_decay: float = 1e-5
    patience: int = 15           # early stop measured in validation checks
    deep_supervision: bool = True
    dsv_weight: float = 0.5      # base weight; deeper (coarser) heads down-weighted
    num_workers: int = 4


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_loss_fn():
    from monai.losses import DiceLoss
    return DiceLoss(sigmoid=True)


def _total_loss(main, aux, target, loss_fn, dsv_w):
    """Deep-supervision loss: main + decreasing weights on the auxiliary heads.
    `aux` is ordered coarsest->finest; coarser heads get smaller weights (a coarse,
    upsampled prediction should not dominate the fine-scale objective)."""
    loss = loss_fn(main, target)
    n = len(aux)
    for i, a in enumerate(aux):
        w = dsv_w * (0.5 ** (n - 1 - i))   # e.g. n=3 -> [0.125, 0.25, 0.5]*dsv_w
        loss = loss + w * loss_fn(a, target)
    return loss


@torch.no_grad()
def volume_dice(model, loader, device, patch=PATCH):
    """Sliding-window Dice/precision/recall over full validation volumes."""
    from monai.inferers import sliding_window_inference
    model.eval()
    dices, precs, recs = [], [], []
    predictor = lambda x: model(x)[0]  # main output only
    for batch in loader:
        img = batch["image"].to(device)
        lbl = (batch["label"] > 0).float().to(device)
        logits = sliding_window_inference(img, patch, sw_batch_size=2, predictor=predictor, overlap=0.25)
        pred = (torch.sigmoid(logits) > 0.5).float()
        tp = (pred * lbl).sum().item()
        fp = (pred * (1 - lbl)).sum().item()
        fn = ((1 - pred) * lbl).sum().item()
        s = 1e-6
        dices.append((2 * tp + s) / (2 * tp + fp + fn + s))
        precs.append((tp + s) / (tp + fp + s))
        recs.append((tp + s) / (tp + fn + s))
    return float(np.mean(dices)), float(np.mean(precs)), float(np.mean(recs)), dices


def train_one_fold(attention_type, fold, dicts, cfg: Cfg3D, device, weights_dir: Path) -> Dict:
    torch.manual_seed(fold)
    np.random.seed(fold)

    train_dicts, val_dicts = kfold_split(dicts, cfg.k_folds, fold)
    train_loader, val_loader = make_loaders(train_dicts, val_dicts, cfg.batch_size,
                                            num_workers=cfg.num_workers)

    model = build_model3d(attention_type, base=cfg.base,
                          deep_supervision=cfg.deep_supervision).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    loss_fn = dice_loss_fn()

    best_dice, best_state, no_improve = -1.0, None, 0
    history = []
    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            img = batch["image"].to(device)
            lbl = (batch["label"] > 0).float().to(device)
            opt.zero_grad()
            main, aux = model(img)
            loss = _total_loss(main, aux, lbl, loss_fn, cfg.dsv_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()  # cosine annealing, once per epoch

        if (epoch + 1) % cfg.val_interval == 0:
            vd, vp, vr, _ = volume_dice(model, val_loader, device)
            history.append({"epoch": epoch, "val_dice": vd, "val_prec": vp, "val_rec": vr})
            print(f"  [{str(attention_type):8s} fold{fold}] epoch {epoch+1:3d} "
                  f"val_dice={vd:.4f} best={max(best_dice, vd):.4f} "
                  f"lr={opt.param_groups[0]['lr']:.1e}")
            if vd > best_dice:
                best_dice, best_state, no_improve = vd, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    print(f"  early stop at epoch {epoch+1}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    vd, vp, vr, per_vol = volume_dice(model, val_loader, device)
    ckpt = weights_dir / f"{attention_type}_fold{fold}.pt"
    torch.save(best_state, ckpt)
    return {"attention_type": str(attention_type), "fold": fold,
            "val_dice": vd, "val_prec": vp, "val_rec": vr,
            "per_volume_dice": per_vol, "num_params": count_parameters(model),
            "weights_path": str(ckpt), "history": history}


def run_kfold(cfg: Cfg3D) -> Dict:
    device = get_device()
    print(f"Device: {device}")
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"; weights_dir.mkdir(exist_ok=True)
    dicts = get_data_dicts(cfg.task_dir)
    print(f"{len(dicts)} labeled volumes; {cfg.k_folds}-fold CV; "
          f"variants={[str(a) for a in cfg.attention_types]}")

    runs = []
    for att in cfg.attention_types:
        for fold in range(cfg.k_folds):
            print(f"=== {att} | fold {fold} ===")
            r = train_one_fold(att, fold, dicts, cfg, device, weights_dir)
            runs.append(r)
            print(f"  -> fold Dice {r['val_dice']:.4f} prec {r['val_prec']:.4f} "
                  f"rec {r['val_rec']:.4f} params {r['num_params']:,}")
            json.dump({"config": asdict(cfg), "runs": runs},
                      open(out_dir / "results_3d.json", "w"), indent=2)

    # Aggregate: mean +/- std over folds per variant
    print("\n===== 3D k-fold summary (mean ± std over folds) =====")
    for att in cfg.attention_types:
        ds = [r["val_dice"] for r in runs if r["attention_type"] == str(att)]
        print(f"  {str(att):8s} Dice {np.mean(ds):.4f} ± {np.std(ds):.4f}")
    print(f"\nSaved to {out_dir/'results_3d.json'}")
    return {"config": asdict(cfg), "runs": runs}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default="./decathlon_data/Task09_Spleen")
    ap.add_argument("--out-dir", default="./results_3d")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--base", type=int, default=16)
    args = ap.parse_args()
    cfg = Cfg3D(task_dir=args.task_dir, out_dir=args.out_dir, k_folds=args.folds,
               max_epochs=args.epochs, base=args.base)
    run_kfold(cfg)
