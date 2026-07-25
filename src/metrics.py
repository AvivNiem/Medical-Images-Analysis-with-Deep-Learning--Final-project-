"""
metrics.py
----------
Evaluation metrics matching those reported in the original Attention U-Net paper
(Table 1): Dice score (DSC), precision, and recall. All operate on binarized
predictions (threshold 0.5 on the sigmoid probability) and return per-sample
values so callers can aggregate mean +/- std across a test set, as the paper does.

A small `smooth` term keeps metrics well-defined on empty ground-truth slices
(background-only slices where the spleen is absent).
"""

from typing import Dict

import numpy as np
import torch


@torch.no_grad()
def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Per-sample Dice similarity coefficient. Returns shape (B,)."""
    preds = _binarize(logits).reshape(logits.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)
    return (2 * inter + smooth) / (denom + smooth)


@torch.no_grad()
def precision_recall(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6):
    """Per-sample precision and recall. Returns two tensors of shape (B,)."""
    preds = _binarize(logits).reshape(logits.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall


class MetricAccumulator:
    """Accumulates per-sample metrics across batches and reports mean +/- std,
    mirroring the paper's reporting format (e.g. '0.840 +/- 0.087').

    By default, background-only slices (empty ground truth) are excluded from the
    Dice average, because Dice on a correctly-predicted empty mask is trivially
    ~1.0 and would inflate the score; they are still useful for inspecting
    false positives, so their false-positive pixel counts are tracked separately."""

    def __init__(self, exclude_empty_gt: bool = True):
        self.exclude_empty_gt = exclude_empty_gt
        self.dice, self.precision, self.recall = [], [], []
        self.empty_gt_false_positive_rate = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        d = dice_score(logits, targets).cpu().numpy()
        p, r = precision_recall(logits, targets)
        p, r = p.cpu().numpy(), r.cpu().numpy()
        has_gt = (targets.reshape(targets.size(0), -1).sum(dim=1) > 0).cpu().numpy()

        preds = _binarize(logits).reshape(logits.size(0), -1).cpu().numpy()
        for i in range(len(d)):
            if has_gt[i]:
                self.dice.append(d[i])
                self.precision.append(p[i])
                self.recall.append(r[i])
            else:
                self.empty_gt_false_positive_rate.append(preds[i].mean())
                if not self.exclude_empty_gt:
                    self.dice.append(d[i])
                    self.precision.append(p[i])
                    self.recall.append(r[i])

    def summary(self) -> Dict[str, float]:
        def ms(x):
            return (float(np.mean(x)), float(np.std(x))) if len(x) else (float("nan"), float("nan"))

        dice_m, dice_s = ms(self.dice)
        prec_m, prec_s = ms(self.precision)
        rec_m, rec_s = ms(self.recall)
        fp_m, _ = ms(self.empty_gt_false_positive_rate)
        return {
            "dice_mean": dice_m, "dice_std": dice_s,
            "precision_mean": prec_m, "precision_std": prec_s,
            "recall_mean": rec_m, "recall_std": rec_s,
            "empty_gt_fp_rate": fp_m,
            "n_foreground_slices": len(self.dice),
        }

    def format_row(self, name: str) -> str:
        s = self.summary()
        return (f"{name:24s} "
                f"DSC {s['dice_mean']:.3f}+/-{s['dice_std']:.3f} | "
                f"Prec {s['precision_mean']:.3f} | "
                f"Rec {s['recall_mean']:.3f} | "
                f"emptyFP {s['empty_gt_fp_rate']:.4f}")
