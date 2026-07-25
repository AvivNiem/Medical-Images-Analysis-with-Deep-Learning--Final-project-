# Context-Gated Channel Attention for Attention U-Net

**Course project — Medical Images Processing with Deep Learning (336033)**
**Base paper:** Oktay et al., *"Attention U-Net: Learning Where to Look for the
Pancreas,"* MIDL 2018 ([arXiv:1804.03999](https://arxiv.org/abs/1804.03999))

## Extension summary

The original Attention U-Net gates skip-connection features using a purely
*spatial* attention coefficient, conditioned on a coarse-scale gating signal `g`.
It never reweights *which feature channels* matter. We extend the attention gate
with a **channel attention branch that is also conditioned on `g`** (not just on
the local features, as in standard self-attention modules like CBAM/SE-Net),
keeping the paper's core "coarse scale tells fine scale what to look for"
principle intact for both spatial and channel gating.

Four model variants are compared, all built from one shared U-Net implementation
(`src/models.py`, `attention_type` argument):

| `attention_type` | Model |
|---|---|
| `None` | Vanilla U-Net |
| `"spatial"` | Attention U-Net (original, reproduced) |
| `"cbam"` | Attention U-Net + naive CBAM bolt-on (control baseline, reproduces the ASCU-Net-style combination) |
| `"hybrid"` | Attention U-Net + **context-gated channel attention (our extension)** |

See `docs/reference_notes.md` for how the implementation was cross-checked
against the official reference code.

## Dataset

**Medical Segmentation Decathlon — Task09_Spleen**: 41 labeled 3D abdominal CT
volumes with binary spleen masks. Chosen to stay close to the original paper's
CT-organ domain while being small enough for a Colab proof-of-concept (the
paper itself used 3D CT organ segmentation; we use 2D axial slices from the
same imaging domain rather than a full 3D pipeline).

**How to get the data** (no need to download manually — this runs inside the
Colab notebook / `src/data.py`):

```bash
pip install monai nibabel scikit-image
python -m src.data --root ./decathlon_data --out ./slices_2d --image-size 128
```

This uses MONAI's `DecathlonDataset` helper to download and extract the
official archive, then slices the 3D volumes into cached 2D axial `.npy` pairs
(soft-tissue HU-windowed, normalized, resized). Patients are split into
train/val/test at the **patient level** (`patient_level_split` in `src/data.py`)
to avoid leaking neighbouring slices of the same patient across splits.

## Project structure

```
attention_unet_spleen/
├── README.md
├── requirements.txt
├── docs/
│   └── reference_notes.md      # cross-check notes vs. official AG implementation
├── src/
│   ├── layers.py                # ConvBlock/Down/Up + all attention gate variants
│   ├── models.py                # configurable AttentionUNet2D (all 4 variants)
│   ├── data.py                  # dataset download + 2D slice preprocessing
│   ├── losses.py                # Soft Dice / BCE+Dice losses
│   ├── metrics.py               # Dice/precision/recall + empty-slice false positives
│   ├── train.py                 # training loop, multi-seed experiment runner
│   ├── stats.py                 # per-variant aggregation + paired Wilcoxon test
│   └── visualize.py             # training curves, overlays, attention maps, failures
└── notebooks/
    └── main_colab.ipynb          # end-to-end driver notebook (run top to bottom)
```

## How to run

Open `notebooks/main_colab.ipynb` in Google Colab (GPU runtime), point the setup
cell at this folder (via GitHub clone or Drive mount), and run top to bottom. The
notebook installs deps, shape-checks all variants, downloads + preprocesses the
data, trains all four variants across seeds, prints the paper-style results table
with paired significance tests, and produces all qualitative figures.

## Status

Implementation complete and ready to run on Colab. Numpy-only logic (metrics,
statistics, patient-level split) unit-tested offline; all modules pass syntax
checks; the torch forward-pass shape test runs as the notebook's first code cell
(torch could not be installed in the offline dev sandbox). The written report is
the remaining deliverable.
