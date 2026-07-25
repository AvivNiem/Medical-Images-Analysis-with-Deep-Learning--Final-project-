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


def build_loss(name: str = "dice") -> nn.Module:
    """Factory: 'dice' (paper default) or 'bce_dice' (more stable default)."""
    if name == "dice":
        return SoftDiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss '{name}' (expected 'dice' or 'bce_dice')")
