# Low-data focused experiment — n = 8 training patients

Faithful 2D spleen pipeline (deep supervision + grid attention + affine aug +
z-score). Fixed held-out test set; **8 seeds** (each seed = a different 8-patient
training subset, identical across variants); per-slice Dice pooled across seeds
(8 seeds x ~248 foreground test slices ≈ 2,000 paired samples per comparison).

## Mean Dice (8 seeds x slices)

| Variant | Dice |
|---|---|
| U-Net (plain) | 0.7835 |
| Attention U-Net (paper's model) | 0.7931 |
| **AG + CBAM** | **0.8770** |
| **Hybrid (ours)** | **0.8241** |

## Pairwise paired Wilcoxon (median diff A−B, p-value)

| A | B | median diff | p | sig |
|---|---|---|---|---|
| U-Net | Attention U-Net | −0.0037 | 0.0008 | *** |
| U-Net | AG+CBAM | −0.0303 | <0.0001 | *** |
| U-Net | Hybrid (ours) | −0.0085 | <0.0001 | *** |
| Attention U-Net | AG+CBAM | −0.0145 | <0.0001 | *** |
| Attention U-Net | Hybrid (ours) | −0.0074 | <0.0001 | *** |
| AG+CBAM | Hybrid (ours) | +0.0146 | <0.0001 | *** |

(A−B negative ⇒ B is better.)

## Key results vs. the paper's model (Attention U-Net)

- **AG+CBAM significantly beats the paper's model**: 0.877 vs 0.793, +0.084 mean
  (median +0.0145, p<0.0001).
- **Our context-gated Hybrid also significantly beats the paper's model**:
  0.824 vs 0.793, +0.031 mean (median +0.0074, p<0.0001).
- AG+CBAM is the strongest overall; our Hybrid is second but still improves on the
  paper's spatial-only attention.

## Interpretation

In the low-data regime (n=8 patients), adding channel attention to the Attention
U-Net's skip-connection gating significantly improves segmentation over the paper's
spatial-only attention gate — for both the CBAM variant and our context-gated
hybrid. This contrasts with the full-data result (see
`results_faithful2d/results_faithful2d_summary.md`), where the same additions give
no significant improvement, and extends the paper's own thesis that attention's
benefit concentrates when training data is scarce.

Caveat: per-slice samples from the same patient are not fully independent, so the
p-values overstate certainty; we therefore emphasize effect sizes (+0.03 to +0.08
Dice), which are substantial. Protocol note: this sweep used a fixed test set with
multiple seeds (appropriate for a data-size study), whereas the full-data result
used 5-fold cross-validation.
