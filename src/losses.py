"""
losses.py
---------
Loss functions for binary 2D spleen segmentation.

The original Attention U-Net paper trains with the Sorensen-Dice loss because it
is "less sensitive to class imbalance" - important here, since the spleen occupies
a small fraction of each CT slice. We provide:

  - SoftDiceLoss  : the paper's choice, differentiable soft Dice over the sigmoid
                    probabilities.
  - BCEDiceLoss   : Dice + binary cross-entropy, a common robust default that
                    tends to stabilise early training (BCE gives a well-behaved
                    gradient when Dice is near-degenerate on empty/near-empty
                    masks). Selectable so we can report either; default matches
                    the paper (pure Dice) unless overridden.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Differentiable soft Dice loss on the foreground channel.

    Operates on raw logits (applies sigmoid internally). `smooth` avoids division
    by zero and stabilises the gradient on empty masks."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice loss."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


class FocalLoss(nn.Module):
    """Binary focal loss on logits.

    Focal loss down-weights easy, well-classified pixels via the (1-pt)^gamma
    factor, so the overwhelming background majority stops dominating the gradient.
    Unlike plain BCE - which on highly imbalanced small-organ slices collapses to
    an all-background prediction - focal loss keeps a useful foreground gradient.
    `alpha` up-weights the (rare) foreground class."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # BCE per pixel, computed in a numerically stable way:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """Sum of soft Dice and focal loss - a stable, imbalance-robust default for
    small-organ segmentation. Dice optimises overlap directly; focal supplies a
    well-behaved per-pixel gradient that prevents the all-background collapse that
    pure Dice can fall into early in training on tiny foreground regions."""

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0,
                 alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + \
            self.focal_weight * self.focal(logits, targets)


def build_loss(name: str = "dice_focal") -> nn.Module:
    """Factory:
      - 'dice_focal' : Dice + focal (stable default, robust to class imbalance)
      - 'dice'       : pure Sorensen-Dice (the original paper's choice; can be
                       unstable on small 2D foreground regions)
      - 'bce_dice'   : Dice + plain BCE (provided for comparison; note BCE can
                       collapse to all-background under heavy imbalance)
    """
    if name == "dice_focal":
        return DiceFocalLoss()
    if name == "dice":
        return SoftDiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss '{name}' (expected 'dice_focal', 'dice', or 'bce_dice')")
