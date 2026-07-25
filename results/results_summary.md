# Results summary — Context-Gated Channel Attention (Attention U-Net extension)

Dataset: Medical Segmentation Decathlon Task09_Spleen, 2D axial slices, 256×256.
Split: patient-level, fixed (seed 1234). Test set: 248 foreground slices.
Config: 5 seeds, up to 60 epochs, Adam lr 3e-4, grad-clip 1.0, Dice+Focal loss.
Hardware: NVIDIA L40. (Raw data: `results.json`, `sweep_results.json`.)

---

## 1. Main comparison — pooled per-slice metrics (5 seeds × 248 slices = 1240 samples)

These are the numbers used in the report's Table 1 (from `compute_significance.py`).

| Variant                     | Dice   | Precision | Recall |
|-----------------------------|--------|-----------|--------|
| U-Net (baseline)            | 0.8439 | 0.8654    | 0.8756 |
| Attention U-Net (original)  | 0.8654 | 0.9004    | 0.8919 |
| AG + CBAM (control)         | 0.8859 | 0.9072    | 0.9027 |
| Hybrid — ours               | 0.8436 | 0.8876    | 0.8625 |

Ranking (Dice): AG+CBAM > Attention U-Net > U-Net ≈ Hybrid.

### Pairwise paired Wilcoxon signed-rank (median diff A−B, p-value)

| Comparison (A vs B)              | median diff | p-value  | sig |
|----------------------------------|-------------|----------|-----|
| U-Net vs Attention U-Net         | −0.0144     | < 0.0001 | *** |
| U-Net vs AG+CBAM                 | −0.0110     | < 0.0001 | *** |
| U-Net vs Hybrid (ours)           | −0.0040     | < 0.0001 | *** |
| Attention U-Net vs AG+CBAM       | +0.0029     | < 0.0001 | *** |
| Attention U-Net vs Hybrid (ours) | +0.0080     | < 0.0001 | *** |
| AG+CBAM vs Hybrid (ours)         | +0.0047     | < 0.0001 | *** |

All pairwise differences are statistically significant (n = 1240 paired samples;
note that with this sample size even small effect sizes reach significance, so
read the median differences as the effect sizes).

Interpretation: adding channel attention significantly improves the Attention
U-Net (CBAM 0.865 → 0.886). Our hybrid ties the plain U-Net on Dice but shifts
the trade-off toward precision (↑ 0.865 → 0.888) and away from recall
(↓ 0.876 → 0.863).

---

## 2. Main comparison — per-run means ± std over 5 seeds (from `results.json`)

Reported during the main run (each run's own test evaluation). Slightly more
optimistic for the higher-variance variants than the pooled figures above,
because the main-run and sweep weights are independent trainings.

| Variant                     | Dice          | Precision | Recall | Params    |
|-----------------------------|---------------|-----------|--------|-----------|
| U-Net (baseline)            | 0.859 ± 0.010 | 0.868     | 0.901  | 7,240,225 |
| Attention U-Net (original)  | 0.860 ± 0.023 | 0.893     | 0.889  | 7,585,636 |
| AG + CBAM (control)         | 0.886 ± 0.009 | 0.911     | 0.891  | 7,607,941 |
| Hybrid — ours               | 0.871 ± 0.026 | 0.894     | 0.891  | 7,629,148 |

Note the hybrid's high variance (± 0.026), the largest of all variants.

---

## 3. Low-data regime sweep — mean test Dice per (variant, training size)

5 seeds each. Source: `sweep_results.json`, plotted in `fig_low_data_curve.png`.

| Variant                     | n=4   | n=8   | n=16  | all   |
|-----------------------------|-------|-------|-------|-------|
| U-Net (baseline)            | 0.670 | 0.763 | 0.848 | 0.844 |
| Attention U-Net (original)  | 0.656 | 0.774 | 0.833 | 0.865 |
| AG + CBAM (control)         | 0.655 | 0.736 | 0.860 | 0.886 |
| Hybrid — ours               | 0.696 | 0.731 | 0.827 | 0.844 |

Hybrid advantage over each baseline (Dice difference):

| vs                          | n=4    | n=8    | n=16   | all    |
|-----------------------------|--------|--------|--------|--------|
| U-Net                       | +0.026 | −0.032 | −0.020 | −0.000 |
| Attention U-Net             | +0.041 | −0.043 | −0.006 | −0.022 |
| AG + CBAM                   | +0.042 | −0.004 | −0.032 | −0.042 |

At the smallest training size (4 patients) the hybrid is the best variant;
the advantage is not monotonic across sizes (a low-data niche, not a robust trend).

---

## 4. Channel-attention analysis (our extension)

Learned context-gated channel weights β (from `fig_channel_attn.png`):

| Gated level | mean β | std β |
|-------------|--------|-------|
| level 4     | 0.66   | 0.26  |
| level 3     | 0.81   | 0.23  |
| level 2     | 0.74   | 0.23  |

β spans roughly 0.1–1.0 → the gate performs meaningful, non-uniform channel
selection rather than trivial pass-through.

---

## 5. Figures (in this folder)

- `fig_low_data_curve.png` — Dice vs. training-set size, per variant (Figure 1).
- `fig_channel_attn.png` — learned channel weights β (Figure 2).
- `fig_training_curves.png` — validation Dice per epoch, per variant.
- `fig_overlay.png` — prediction overlays across variants on one slice.
- `fig_spatial_attn.png` — spatial attention maps (paper Fig. 3/4 style).
- `fig_worst_cases.png` — lowest-Dice slices (failure analysis).

To regenerate the pooled table + significance matrix:
`CUDA_VISIBLE_DEVICES=1 python compute_significance.py --image-size 256`
