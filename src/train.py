"""
train.py
--------
Training pipeline for the four model variants. Designed to be driven either from
the command line or (more commonly) imported and called cell-by-cell in the Colab
notebook.

Key features relevant to the project's scientific goals:
  - `train_one_model()`   : trains a single variant with Adam + Dice loss (paper
                            settings), early stopping on validation Dice, and
                            per-epoch history logging.
  - `run_experiment()`    : trains ALL requested variants across MULTIPLE random
                            seeds, so we can report mean +/- std and run paired
                            significance tests between variants - the same
                            statistical rigor the original paper uses (it reports
                            p-values). Multi-seed is what makes the comparison
                            credible rather than a single-run fluke.
  - deterministic seeding : per-run reproducibility.
  - device-agnostic       : runs on Colab GPU if available, CPU otherwise; the
                            optional external GPU server changes nothing but speed.
"""

import copy
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SpleenSliceDataset, patient_level_split
from .losses import build_loss
from .metrics import MetricAccumulator, dice_score
from .models import build_model, count_parameters


@dataclass
class TrainConfig:
    """All training hyperparameters in one place, serialized alongside results
    for reproducibility."""
    slice_dir: str = "./slices_2d"
    attention_types: List[Optional[str]] = field(
        default_factory=lambda: [None, "spatial", "cbam", "hybrid"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    base_channels: int = 32
    image_size: int = 128
    batch_size: int = 8
    max_epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0            # max gradient norm; guards against divergence
    loss_name: str = "dice_focal"     # Dice + focal: stable & imbalance-robust. Pure
                                      #   'dice' can collapse to all-background early on
                                      #   tiny 2D foreground; plain BCE collapses worse.
    patience: int = 12                # early-stopping patience (epochs w/o val improvement)
    num_workers: int = 2
    out_dir: str = "./results"
    val_frac: float = 0.15
    test_frac: float = 0.2
    train_size: Optional[int] = None  # if set, subsample this many training PATIENTS
                                      #   (for the low-data-regime sweep). None = all.


def set_seed(seed: int):
    """Seed python/numpy/torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loaders(cfg: TrainConfig, seed: int):
    """Builds train/val/test DataLoaders with a fixed patient-level split.

    NOTE: the split is seeded independently of the model seed so that all model
    variants and all model seeds see the SAME patient split - otherwise a variant
    could look better purely because it got an easier test set. Only weight
    initialisation and augmentation order vary with the model seed."""
    slice_dir = Path(cfg.slice_dir)
    train_ids, val_ids, test_ids = patient_level_split(
        slice_dir, val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=1234)

    # Low-data-regime subsampling: keep only `train_size` training patients.
    # The subset is seeded by the model seed so different seeds see different
    # subsets (giving meaningful variance), but within a seed ALL variants see the
    # SAME subset - so the comparison at each training size is fair. The val/test
    # splits are untouched, so every training size is evaluated on the same test set.
    if cfg.train_size is not None and cfg.train_size < len(train_ids):
        subset_rng = random.Random(1000 + seed)
        train_ids = subset_rng.sample(train_ids, cfg.train_size)

    train_ds = SpleenSliceDataset(slice_dir, train_ids, augment=True)
    val_ds = SpleenSliceDataset(slice_dir, val_ids, augment=False)
    test_ds = SpleenSliceDataset(slice_dir, test_ids, augment=False)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, generator=g, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader, (train_ids, val_ids, test_ids)


@torch.no_grad()
def _validate(model, loader, device) -> float:
    """Mean foreground Dice over a loader (used for early stopping)."""
    model.eval()
    scores = []
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        logits = model(img)
        has_gt = msk.reshape(msk.size(0), -1).sum(dim=1) > 0
        if has_gt.any():
            d = dice_score(logits[has_gt], msk[has_gt])
            scores.extend(d.cpu().numpy().tolist())
    return float(np.mean(scores)) if scores else 0.0


def train_one_model(attention_type: Optional[str], cfg: TrainConfig, seed: int,
                     verbose: bool = True) -> Dict:
    """Train a single variant/seed. Returns a result dict with history, best
    validation Dice, test metrics, param count, and inference time."""
    set_seed(seed)
    device = get_device()

    train_loader, val_loader, test_loader, splits = _make_loaders(cfg, seed)

    model = build_model(attention_type, base_channels=cfg.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = build_loss(cfg.loss_name)

    best_val, best_state, epochs_no_improve = -1.0, None, 0
    history = {"train_loss": [], "val_dice": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses = []
        for img, msk in train_loader:
            img, msk = img.to(device), msk.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), msk)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        val_dice = _validate(model, val_loader, device)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_dice"].append(val_dice)

        if val_dice > best_val:
            best_val, best_state, epochs_no_improve = val_dice, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == cfg.max_epochs - 1):
            print(f"  [{str(attention_type):8s} seed={seed}] "
                  f"epoch {epoch:3d} loss={history['train_loss'][-1]:.4f} "
                  f"val_dice={val_dice:.4f} best={best_val:.4f}")

        if epochs_no_improve >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (no val improvement for {cfg.patience})")
            break

    # Restore best checkpoint and evaluate on the test set.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_on_test(model, test_loader, device)
    inf_time = measure_inference_time(model, device, cfg.image_size)

    return {
        "attention_type": str(attention_type),
        "seed": seed,
        "train_size": cfg.train_size,   # None = full training set
        "best_val_dice": best_val,
        "test_metrics": test_metrics,
        "num_params": count_parameters(model),
        "inference_time_s": inf_time,
        "history": history,
        "splits": {k: v for k, v in zip(["train", "val", "test"], splits)},
        "best_state": best_state,   # kept in-memory for optional attention-map viz
    }


@torch.no_grad()
def evaluate_on_test(model, loader, device) -> Dict[str, float]:
    """Full paper-style metric summary on the test set."""
    model.eval()
    acc = MetricAccumulator(exclude_empty_gt=True)
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        acc.update(model(img), msk)
    return acc.summary()


@torch.no_grad()
def measure_inference_time(model, device, image_size: int, n: int = 20) -> float:
    """Mean forward-pass time for a single image, matching the paper's reported
    'Inference Time' column (they use a fixed input size)."""
    model.eval()
    x = torch.randn(1, 1, image_size, image_size, device=device)
    for _ in range(3):  # warmup
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def run_experiment(cfg: TrainConfig) -> Dict:
    """Train every (variant x seed) combination, save results to disk, and return
    the aggregated results dict. This is the main entry point called by the
    notebook."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for att in cfg.attention_types:
        for seed in cfg.seeds:
            print(f"=== Training attention_type={att} seed={seed} ===")
            all_runs.append(_train_and_save(att, cfg, seed, weights_dir))

    results = {"config": asdict(cfg), "runs": all_runs}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / 'results.json'}")
    print(f"Saved model weights to {weights_dir}/")
    return results


def _train_and_save(att, cfg: TrainConfig, seed: int, weights_dir: Path) -> Dict:
    """Train one (variant, seed[, train_size]) run, persist its best checkpoint to
    disk, and return a JSON-serializable run dict (weights replaced by a path).
    Shared by run_experiment and run_low_data_sweep."""
    result = train_one_model(att, cfg, seed)

    # Checkpoint filename encodes variant + seed + training size so the low-data
    # sweep's many runs don't overwrite one another.
    tag = f"{att}_seed{seed}_n{cfg.train_size if cfg.train_size is not None else 'all'}"
    ckpt_path = weights_dir / f"{tag}.pt"
    torch.save(result["best_state"], ckpt_path)

    serializable = {k: v for k, v in result.items() if k != "best_state"}
    serializable["weights_path"] = str(ckpt_path)

    row = result["test_metrics"]
    print(f"  -> test DSC {row['dice_mean']:.3f}+/-{row['dice_std']:.3f} "
          f"prec {row['precision_mean']:.3f} rec {row['recall_mean']:.3f} "
          f"params {result['num_params']:,}")
    return serializable


def run_low_data_sweep(cfg: TrainConfig, train_sizes: List[Optional[int]]) -> Dict:
    """Low-data-regime experiment: train every (variant x seed) at each training
    set size in `train_sizes` (a value of None means the full training set). This
    reproduces the original paper's key analysis - whether attention's benefit
    grows as training data shrinks - and is where our context-gated channel gate
    is hypothesised to help most.

    Saves everything to results/sweep_results.json and returns the results dict.
    """
    from dataclasses import replace

    out_dir = Path(cfg.out_dir)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for ts in train_sizes:
        cfg_ts = replace(cfg, train_size=ts)
        label = ts if ts is not None else "all"
        for att in cfg.attention_types:
            for seed in cfg.seeds:
                print(f"=== train_size={label}  attention_type={att}  seed={seed} ===")
                all_runs.append(_train_and_save(att, cfg_ts, seed, weights_dir))

    results = {"config": asdict(cfg), "train_sizes": train_sizes, "runs": all_runs}
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sweep results to {out_dir / 'sweep_results.json'}")
    return results


def load_trained_model(results: Dict, attention_type, seed: int, cfg: TrainConfig,
                        device=None, train_size=None):
    """Reload a trained model saved by run_experiment / run_low_data_sweep (matched
    by variant + seed + training size), so downstream cells don't have to retrain.
    Returns the model in eval mode."""
    device = device or get_device()
    att_str = str(attention_type)
    ckpt = None
    for run in results["runs"]:
        if (run["attention_type"] == att_str and run["seed"] == seed
                and run.get("train_size", None) == train_size):
            ckpt = run.get("weights_path")
            break
    if ckpt is None:
        raise ValueError(f"No saved weights for attention_type={att_str}, "
                         f"seed={seed}, train_size={train_size}")

    model = build_model(attention_type, base_channels=cfg.base_channels).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", default="./slices_2d")
    parser.add_argument("--out-dir", default="./results")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--loss", default="dice", choices=["dice", "bce_dice"])
    args = parser.parse_args()

    cfg = TrainConfig(slice_dir=args.slice_dir, out_dir=args.out_dir,
                       max_epochs=args.epochs, seeds=args.seeds, loss_name=args.loss)
    run_experiment(cfg)
