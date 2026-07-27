# 3D Faithful Reproduction — Attention U-Net + channel-attention variants

This folder (`spleen_3d/`, on the `spleen-3d` branch) is the **rigorous, faithful
reproduction** of the study: a 3D Attention U-Net that reproduces the original
paper's setup as closely as practical, changing **only** the attention module
between the compared variants.

## What is faithful to the paper here (vs. the 2D proof-of-concept on `main`)

| Aspect | 2D proof-of-concept (`main`) | 3D reproduction (this folder) |
|---|---|---|
| Dimensionality | 2D axial slices | **3D volumes / patches** |
| Attention gate | our 2D port | **paper's gate ported to 3D** (from `grid_attention_layer.py`) |
| Deep supervision | omitted | **included** (paper's `*_dsv`) |
| Loss | Dice + Focal (needed for 2D stability) | **plain Dice** (paper's; stable via balanced patch sampling) |
| Augmentation | flips + 90° rot | **affine (rot/scale) + flips** (paper-style) |
| Evaluation | single split × 5 seeds | **k-fold cross-validation** (paper's protocol) |
| Metrics | Dice / precision / recall | Dice / precision / recall (+ surface distance, planned) |

The **only** difference between the four models is the `attention_type` argument:
`None` (U-Net), `"spatial"` (Attention U-Net), `"cbam"` (AG+CBAM control),
`"hybrid"` (AG + context-gated channel attention, ours).

## Files
- `layers3d.py`  — 3D blocks: paper's grid attention gate, CBAM-3D, our context-gated
  channel gate, and one configurable `AttentionGate3D`.
- `model3d.py`   — configurable `AttentionUNet3D` with deep supervision.
- `data3d.py`    — MONAI-based 3D pipeline (spacing, HU window, patch sampling,
  affine/flip augmentation) + patient-level k-fold splitter.
- `train3d.py`   — Dice loss + deep supervision, sliding-window validation, k-fold runner.
- `run3d.py`     — entry point: shape check + `--quick` smoke test / full k-fold run.

## How to run (on the GPU server)

```bash
cd ~/Medical-Images-Analysis-with-Deep-Learning--Final-project-
git fetch && git checkout spleen-3d && git pull
pip install -r requirements.txt          # ensure monai is installed

# 1) quick end-to-end smoke test (fold 0, 2 variants, tiny net):
CUDA_VISIBLE_DEVICES=1 python -m spleen_3d.run3d --quick

# 2) full reported run (5-fold CV, all 4 variants):
CUDA_VISIBLE_DEVICES=1 python -m spleen_3d.run3d
```

The data (Task09_Spleen) is reused from the 2D work: `download_spleen_dataset`
already fetched it into `decathlon_data/`. Point `--task-dir` at
`decathlon_data/Task09_Spleen` if needed.

Outputs go to `results_3d/` (per-fold metrics in `results_3d.json`, checkpoints in
`results_3d/weights/`).
```
