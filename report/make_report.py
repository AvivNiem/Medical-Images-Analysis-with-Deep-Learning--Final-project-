"""
make_report.py — generate the final 6-page PDF report with reportlab.
Embeds the real result numbers, tables, and figures from results/.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, HRFlowable)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "final_report.pdf")

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.3, leading=12.6,
                      alignment=TA_JUSTIFY, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=12.5, spaceBefore=8,
                    spaceAfter=4, textColor=colors.HexColor("#1a3c6e"))
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10.5, spaceBefore=5,
                    spaceAfter=3, textColor=colors.HexColor("#25507f"))
title = ParagraphStyle("title", parent=styles["Title"], fontSize=17, leading=21,
                       alignment=TA_CENTER, textColor=colors.HexColor("#12294d"))
sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
                     leading=14)
cap = ParagraphStyle("cap", parent=styles["Normal"], fontSize=8, leading=10,
                     alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                     spaceBefore=2, spaceAfter=8)

S = []
def P(t, s=body): S.append(Paragraph(t, s))
def SP(h=4): S.append(Spacer(1, h))


def fig(name, width=13*cm, caption=None):
    path = os.path.join(RESULTS, name)
    if os.path.exists(path):
        from PIL import Image as PILImage
        w, h = PILImage.open(path).size
        img = Image(path, width=width, height=width*h/w)
        img.hAlign = "CENTER"
        S.append(img)
        if caption:
            S.append(Paragraph(caption, cap))


# ---------------- Title ----------------
P("Context-Gated Channel Attention:<br/>Extending Attention U-Net with Top-Down Channel Selection", title)
SP(10)
S.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#1a3c6e"),
                    spaceBefore=2, spaceAfter=8, hAlign="CENTER"))
P("Medical Images Processing with Deep Learning (336033) — Final Project", sub)
SP(6)
P("<b>Aviv Niemann</b> (ID: __________) &nbsp;&nbsp;·&nbsp;&nbsp; <b>______________</b> (ID: __________)", sub)
SP(10)
P("<b>Selected paper:</b> O. Oktay et al., \"Attention U-Net: Learning Where to Look "
  "for the Pancreas,\" MIDL 2018 (arXiv:1804.03999)", sub)
SP(6)
P("<b>Proposed extension:</b> a context-gated channel-attention branch added to the "
  "attention gate, conditioned on the same coarse-scale gating signal the original "
  "spatial gate uses — extending the paper's top-down \"where to look\" mechanism "
  "from spatial locations to feature channels.", sub)
SP(10)
S.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#999999"),
                    spaceBefore=2, spaceAfter=8))

# ---------------- 1. Introduction ----------------
P("1. Introduction", h1)
P("<b>Summary of the paper.</b> The Attention U-Net augments the standard U-Net with "
  "<i>attention gates</i> (AGs) on the skip connections. Each gate takes the fine-scale "
  "encoder features <i>x</i> that would be concatenated into the decoder together with a "
  "coarser-scale <i>gating signal g</i> from the deeper decoder path, and produces a "
  "per-pixel spatial attention coefficient &alpha; &isin; [0,1]. Multiplying <i>x</i> by "
  "&alpha; suppresses background responses before they reach the decoder, so the network "
  "learns to focus on the target organ without an external localization module. The gates "
  "are additive, trained end-to-end, and add few parameters. On abdominal CT the authors "
  "show AGs improve pancreas segmentation — notably increasing recall — with the benefit "
  "most pronounced when training data is scarce.")
P("<b>Problem and relevance.</b> Automated organ segmentation in CT is clinically valuable "
  "but hard for small, low-contrast, shape-variable structures. Prior state of the art relied "
  "on multi-stage cascades (localize, then segment), which are redundant and complex. The "
  "attention gate folds localization into a single end-to-end model for negligible cost, and "
  "its attention maps offer a degree of interpretability.")
P("<b>Motivation.</b> The gate is elegant and widely used, but has a clear limitation: its "
  "attention is purely <i>spatial</i> — it decides <i>where</i> to look but applies the same "
  "scalar across all feature channels, never deciding <i>which</i> channels are relevant. "
  "Channel-attention modules (SE-Net, CBAM) address channel selection, but via pure "
  "self-attention over the feature map's own statistics, discarding the paper's core idea of "
  "top-down contextual gating. This gap is a natural place to extend the method.")

# ---------------- 2. Proposed Extension ----------------
P("2. Proposed Extension", h1)
P("<b>Description.</b> We add a second gating branch that produces a per-channel weight "
  "&beta; &isin; [0,1]<super>C</super> applied alongside the spatial coefficient, so the gated "
  "skip output becomes <i>W(x &middot; &alpha; &middot; &beta;)</i>. Crucially &beta; is computed from "
  "<b>both</b> the local features <i>x</i> and the coarse gating signal <i>g</i>, mirroring "
  "the spatial gate: &beta; = &sigma;(W<sub>out</sub> ReLU(W<sub>x</sub> GAP(x) + "
  "W<sub>g</sub> GAP(g))), where GAP is global average pooling. The channel gate is thus "
  "<i>context-gated</i>: the coarse decoder context helps decide which channels matter, just "
  "as it already decides which locations matter. It is initialized to pass all channels "
  "(&beta; &asymp; 1) so it cannot suppress useful features before learning.")
P("<b>Gap addressed and hypothesis.</b> The original AG collapses channel information into a "
  "single spatial scalar. We hypothesize that different channels carry different, "
  "location-dependent relevance — especially for small, low-contrast organs — and that letting "
  "top-down context modulate channels improves the precision/recall trade-off. We further "
  "hypothesize the advantage over both the baseline and a naïve channel-attention bolt-on "
  "<b>grows as training data shrinks</b>, the regime the original paper highlights as most "
  "favorable to attention.")
P("<b>Alternatives considered.</b> (i) A standard CBAM block on the AG — its channel attention "
  "is pure self-attention and ignores <i>g</i>; we retain it as a <i>control</i> to isolate the "
  "value of context-conditioning. (ii) Squeeze-and-Excitation in the encoder — same objection, "
  "not tied to skip gating. (iii) Residual/highway connections around the gate — the original "
  "authors reported no benefit, so we did not pursue it.")

# ---------------- 3. Methodology ----------------
P("3. Methodology", h1)
P("<b>Models.</b> All four variants are built from one configurable 2D U-Net so the only "
  "difference is the gating module: <b>U-Net</b> (no gating); <b>Attention U-Net</b> (original "
  "spatial gate &alpha;(x,g)); <b>AG+CBAM</b> (spatial gate + CBAM channel/spatial "
  "self-attention, the control); and <b>Hybrid, ours</b> (spatial gate &alpha;(x,g) and "
  "context-gated channel gate &beta;(x,g)). Following the paper, the shallowest skip is left "
  "ungated; the spatial gate reproduces the authors' reference implementation.")
P("<b>Data and preprocessing.</b> Medical Segmentation Decathlon Task09_Spleen — 41 labeled 3D "
  "abdominal CT volumes with binary spleen masks, chosen to stay in the paper's CT small-organ "
  "domain while remaining a feasible proof-of-concept. Volumes are windowed to a soft-tissue HU "
  "range, normalized, sliced to 2D axial slices and resized to 256&times;256; all spleen slices "
  "plus a sample of background slices are kept. Patients are split train/val/test at the "
  "<b>patient level</b> (never per slice) to prevent leakage; the same test set is used for every "
  "variant and training size.")
P("<b>Training.</b> Adam (lr 3&times;10<super>-4</super>), batch 8, up to 60 epochs with early "
  "stopping on validation Dice, gradient clipping, light augmentation, 5 random seeds; we report "
  "mean and a paired Wilcoxon signed-rank test on per-slice Dice. <b>Loss — a stability finding:</b> "
  "the paper uses Dice loss for imbalance robustness, but on our small-foreground 2D slices pure "
  "Dice training was stochastically unstable, intermittently collapsing to an all-background "
  "prediction from which Dice's vanishing gradient could not recover; plain BCE+Dice was worse "
  "under heavy imbalance. We adopted a <b>Dice+Focal</b> compound loss — Dice optimizes overlap, "
  "focal down-weights easy background and preserves a strong foreground gradient — which eliminated "
  "the collapse across all 100 training runs. This instability is an artifact of the small-foreground "
  "2D setting and does not arise in the paper's full-3D regime.")
P("<b>Experiments.</b> (1) main comparison of all four variants at full data; (2) a low-data sweep "
  "training each variant at 4, 8, 16 and all patients; (3) interpretability via spatial attention "
  "maps and the learned channel weights &beta;. <b>Tools:</b> PyTorch, MONAI, nibabel, scikit-image, "
  "SciPy, matplotlib. Full experiments ran on an NVIDIA L40; the submitted Colab notebook reproduces "
  "the whole pipeline and defaults to a reduced proof-of-concept configuration.")

# ---------------- 4. Results ----------------
P("4. Results and Analysis", h1)
P("<b>Main comparison (full data, 5 seeds; pooled per-slice metrics over 1240 foreground "
  "slices).</b> All pairwise differences are statistically significant (paired Wilcoxon, "
  "p &lt; 0.0001).")

data = [
    ["Variant", "Dice", "Precision", "Recall"],
    ["U-Net (baseline)", "0.844", "0.865", "0.876"],
    ["Attention U-Net (original)", "0.865", "0.900", "0.892"],
    ["AG + CBAM (control)", "0.886", "0.907", "0.903"],
    ["Hybrid — ours", "0.844", "0.888", "0.863"],
]
t = Table(data, colWidths=[6.2*cm, 2.3*cm, 2.6*cm, 2.3*cm], hAlign="CENTER")
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c6e")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef2f8")]),
    ("BACKGROUND", (0,3), (-1,3), colors.HexColor("#dCe8d0")),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#b0b8c4")),
    ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
]))
S.append(t)
S.append(Paragraph("Table 1. Full-data segmentation results. AG+CBAM is best; the original "
                   "Attention U-Net significantly improves over the plain U-Net; our hybrid ties "
                   "the U-Net on Dice while trading recall for precision.", cap))

P("<b>Channel attention significantly improves the paper's method.</b> Both attention variants "
  "beat the plain U-Net, and adding channel attention raises Dice further: the CBAM channel "
  "attention lifts the original Attention U-Net from 0.865 to <b>0.886</b> (p &lt; 0.0001), the "
  "best result overall on every metric. This is the central positive finding — the Attention "
  "U-Net leaves channel selection unexploited, and adding it helps.")
P("<b>Our context-gated variant: a precision/recall trade-off.</b> At full data the hybrid ties "
  "the plain U-Net on Dice (0.844) and is significantly below the original Attention U-Net and "
  "CBAM. However, relative to the U-Net it <i>raises precision</i> (0.865 &rarr; 0.888) while "
  "<i>lowering recall</i> (0.876 &rarr; 0.863): the context-gated channel gate makes the model "
  "more conservative — predicting less spleen but more accurately — rather than more sensitive. "
  "This differs from the original paper's recall-boosting effect and indicates the extra "
  "channel gate is actively reshaping the decision, not merely adding capacity (it adds only "
  "0.6% parameters over the spatial gate).")

fig("fig_low_data_curve.png", width=11.5*cm,
    caption="Figure 1. Low-data regime: test Dice vs. training-set size (mean ± std, 5 seeds). "
            "Our hybrid (red) is the best variant at the smallest training size (4 patients), "
            "consistent with the hypothesis that context-gated channel selection helps most when "
            "data is scarce; the advantage does not persist at larger sizes.")

P("<b>Low-data regime.</b> Figure 1 tests our central hypothesis. At the smallest training set "
  "(4 patients) the hybrid is the top performer (Dice 0.696 vs. 0.670/0.656/0.655 for "
  "U-Net/Attention/CBAM), supporting the idea that a context-conditioned channel gate is most "
  "useful when per-image statistics are unreliable. The advantage is not monotonic — at 8 and 16 "
  "patients the hybrid is no longer ahead — so the evidence is suggestive of a low-data niche "
  "rather than a robust trend.")

fig("fig_channel_attn.png", width=10.5*cm,
    caption="Figure 2. Learned context-gated channel weights β at three gated levels (sorted). "
            "β spans ~0.1–1.0 (means 0.66–0.81, std ≈ 0.23), showing the gate performs meaningful, "
            "non-uniform channel selection rather than trivial pass-through.")

P("<b>Interpretability.</b> Figure 2 shows the learned channel weights &beta; are far from uniform "
  "(they span roughly 0.1–1.0), confirming the gate learns genuine channel selection. Qualitative "
  "prediction overlays and spatial attention maps (in the repository) reproduce the paper's "
  "observation that gates localize the organ; the failure cases are dominated by thin organ tips "
  "and low-contrast boundary slices, where all variants struggle.")

# ---------------- 5. Conclusion ----------------
P("5. Conclusion", h1)
P("<b>Findings.</b> Starting from the hypothesis that context-gated channel attention would improve "
  "the Attention U-Net, we found a more nuanced result. Adding channel attention to the attention "
  "gate significantly improves spleen segmentation — a CBAM-style channel attention raises Dice from "
  "0.865 to 0.886 (p &lt; 0.0001) and is best on every metric. Our specific context-gated variant "
  "does not beat the original at full data; instead it shifts the model toward higher precision and "
  "lower recall, and its clearest benefit is the extreme low-data regime, where it is the best "
  "variant. The learned channel weights confirm the mechanism performs real, non-uniform channel "
  "selection.")
P("<b>Limitations.</b> A 2D proof-of-concept on a single organ at downsampled resolution, with a "
  "small dataset and high seed-to-seed variance for the gated variants; the required Dice+Focal loss "
  "adaptation; and, with 1240 paired samples, statistical significance that reflects small effect "
  "sizes. Results should be read as indicative, not definitive.")
P("<b>Future work.</b> Extend to full 3D volumes and multiple organs; add deep supervision (used in "
  "the original but omitted here); investigate why context-conditioning underperforms simple channel "
  "recalibration at full data while helping at low data; and test on thin, branching structures such "
  "as retinal vessels, where channel-specific selection may matter more.")

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.4*cm, bottomMargin=1.3*cm,
                        leftMargin=1.7*cm, rightMargin=1.7*cm)
doc.build(S)
print("wrote", OUT)
