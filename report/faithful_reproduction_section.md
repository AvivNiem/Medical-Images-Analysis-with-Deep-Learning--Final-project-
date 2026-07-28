# Faithful Reproduction with k-fold Cross-Validation (report section)

*Drop-in section for the final report, covering the more faithful reproduction and
its cross-validated results. Pairs with `results_faithful2d/fig_kfold_box.png`.*

## Methodology — a faithful reproduction

Following feedback that the ideal study reproduces the original method as closely as
practical and changes **only** the architecture, we built a second, more faithful
pipeline. It keeps every non-architectural choice identical across the four variants
and adds the elements our initial proof-of-concept omitted:

- **Grid attention gate** ported directly from the authors' public implementation.
- **Deep supervision**: auxiliary segmentation heads at each decoder scale, jointly
  supervised (the paper's `*_dsv` configuration), with coarser heads down-weighted.
- **Paper-style augmentation**: affine transformations (rotation, scale) plus flips.
- **Intensity normalization** to zero mean / unit standard deviation (per slice),
  matching the paper's N(0,1) normalization.
- **5-fold patient-level cross-validation** (each patient tested exactly once),
  replacing the single train/val/test split.

We attempted a full 3D reproduction first, but 3D training on the small dataset was
unstable to converge within our compute/time budget; we therefore ran the faithful
pipeline on 2D axial slices, where training is reliable. The one deliberate
deviation retained from the proof-of-concept is a Dice+Focal loss (pure Dice is
unstable on small-foreground 2D slices); this is applied identically to all variants.
Only the `attention_type` (none / spatial grid gate / AG+CBAM / context-gated hybrid)
differs between models.

## Results

**Per-fold test Dice (mean ± std over the 5 folds):**

| Variant | Dice |
|---|---|
| U-Net (baseline) | 0.856 ± 0.061 |
| Attention U-Net (original) | 0.869 ± 0.048 |
| AG + CBAM (control) | 0.872 ± 0.034 |
| Hybrid (ours) | 0.852 ± 0.079 |

**Pooled per-slice metrics (over all folds):**

| Variant | Dice | Precision | Recall |
|---|---|---|---|
| U-Net | 0.862 | 0.906 | 0.885 |
| Attention U-Net | 0.870 | 0.908 | 0.895 |
| AG + CBAM | 0.874 | 0.908 | 0.892 |
| Hybrid (ours) | 0.862 | 0.903 | 0.878 |

**Paired significance (Wilcoxon signed-rank on per-slice Dice, median difference, p):**

| Comparison | median diff | p | significant? |
|---|---|---|---|
| U-Net vs Attention U-Net | +0.0000 | 0.047 | marginal (effect ≈ 0) |
| U-Net vs AG+CBAM | −0.0000 | 0.262 | no |
| U-Net vs Hybrid | +0.0005 | 0.145 | no |
| Attention U-Net vs AG+CBAM | +0.0002 | 0.274 | no |
| Attention U-Net vs Hybrid | +0.0031 | <0.001 | yes (tiny) |
| AG+CBAM vs Hybrid | +0.0022 | <0.001 | yes (tiny) |

See `fig_kfold_box.png` for the per-fold distribution: the four variants overlap
almost entirely, with fold-to-fold variation far exceeding any between-variant gap.

## Analysis

Under proper cross-validation, **the attention mechanisms give no meaningful
improvement over a well-trained U-Net on this task.** The best variant (AG+CBAM) is
not significantly better than the baseline (p = 0.26); the original attention gate is
"significant" only technically (p = 0.047, median difference 0.0000); and the only
consistent effect is that our context-gated hybrid is marginally *worse* than the
spatial and CBAM variants — statistically detectable but ≤ 0.003 Dice, i.e.
practically negligible. All effect sizes are ≤ 0.003 Dice.

Two points explain and contextualize this. First, **the faithful recipe strengthened
the baseline**: deep supervision, z-score normalization, and affine augmentation
raised the plain U-Net from ~0.844 (our proof-of-concept) to ~0.862, leaving little
headroom for attention to add value. Second, **the spleen is an easy target**
(large, high-contrast, compact), so all models already reach ~0.86–0.87; attention
gates are designed to help localize small, low-contrast, shape-variable structures
(the paper's pancreas), and that is precisely where our earlier weaker-baseline 2D
experiment did show a benefit (and where the low-data regime favored the hybrid).

We emphasize **effect sizes over p-values**: with thousands of pooled slices even a
0.002 Dice difference can reach statistical significance, but slices from the same
patient are not fully independent, so those p-values overstate certainty. The
honest conclusion is that the differences are negligible in practice.

## Conclusion

A careful, cross-validated reproduction shows that the attention gate's benefit is
**context-dependent and small**: on an easy organ with a strong, well-regularized
baseline it essentially vanishes, and our context-gated channel extension does not
improve on the original — echoing our proof-of-concept finding that channel
attention's value concentrates in harder, low-data regimes rather than easy,
data-rich ones. This negative-but-rigorous result is, we argue, more informative than
a headline accuracy gain: it delineates *when* the paper's mechanism helps and when
it does not.
