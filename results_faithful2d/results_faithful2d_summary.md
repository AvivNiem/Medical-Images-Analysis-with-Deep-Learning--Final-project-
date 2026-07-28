# Faithful 2D reproduction — results

5-fold patient-level cross-validation. Deep supervision + grid attention + affine augmentation + z-score normalization.

## Per-fold Dice (mean ± std over folds)

| Variant | Dice |
|---|---|
| U-Net | 0.8562 ± 0.0606 |
| Attention U-Net | 0.8694 ± 0.0484 |
| AG+CBAM | 0.8720 ± 0.0341 |
| Hybrid(ours) | 0.8522 ± 0.0793 |

## Pooled per-slice metrics

| Variant | Dice | Precision | Recall |
|---|---|---|---|
| U-Net | 0.8624 | 0.9062 | 0.8845 |
| Attention U-Net | 0.8700 | 0.9075 | 0.8945 |
| AG+CBAM | 0.8740 | 0.9079 | 0.8922 |
| Hybrid(ours) | 0.8621 | 0.9032 | 0.8783 |

## Pairwise paired Wilcoxon (median diff A−B, p)

| A | B | median diff | p | sig |
|---|---|---|---|---|
| U-Net | Attention U-Net | +0.0000 | 0.0470 | * |
| U-Net | AG+CBAM | -0.0000 | 0.2620 | ns |
| U-Net | Hybrid(ours) | +0.0005 | 0.1448 | ns |
| Attention U-Net | AG+CBAM | +0.0002 | 0.2743 | ns |
| Attention U-Net | Hybrid(ours) | +0.0031 | 0.0000 | *** |
| AG+CBAM | Hybrid(ours) | +0.0022 | 0.0000 | *** |
