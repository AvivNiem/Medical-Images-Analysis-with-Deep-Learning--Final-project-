# Reference notes: cross-checking against the official Attention U-Net implementation

Source: [ozan-oktay/Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks)
(MIT License, Copyright (c) 2018 Ozan Oktay), cloned locally for reference only —
**not** included in the submitted source code, since we reimplement everything
ourselves in 2D for Colab (see `src/layers.py`, `src/models.py`).

## What we verified against `models/layers/grid_attention_layer.py`

The official `_GridAttentionBlockND` class confirms the additive attention gate is
implemented as:

- `theta`: a conv on `x` with `kernel_size=stride=sub_sample_factor` — i.e. it
  **downsamples x** to the gating signal's coarser resolution using a strided
  convolution (not a plain 1x1 conv as a literal reading of the paper's Eq. 1 might
  suggest — the paper's grid-resampling detail is implemented via this
  strided-conv + resize approach, not by upsampling `g`).
- `phi`: a 1x1 conv on `g`, resized (bilinear/trilinear) to match `theta(x)`'s
  spatial resolution if they don't already match.
- `f = ReLU(theta(x) + phi(g))`, `psi(f)` (1x1 conv to 1 channel), then `Sigmoid`
  to produce the attention coefficients at the coarse resolution.
- The attention map is then **upsampled back to x's original resolution**
  (trilinear in 3D / bilinear in our 2D case) before being multiplied with `x`.
- A final `W` (1x1 conv + BatchNorm) output transform is applied to `x * alpha`.
- The class also supports `dimension=2`, confirming a 2D variant is a natural,
  intended reduction of the method (not something we're improvising).

Our `SpatialAttentionGate2D` + `AttentionBlock2D.output_transform` in
`src/layers.py` reproduces exactly this structure (theta/phi/psi + resize + W),
adapted to plain `nn.Conv2d`/`nn.BatchNorm2d`.

## Where we deliberately depart from the reference repo

- **2D instead of 3D**: the reference repo targets full 3D CT volumes with 3D
  convolutions; we use 2D axial slices for Colab feasibility (see project
  instructions: proof-of-concept on a simplified task, not full-scale 3D training).
- **No deep supervision**: the original paper uses deep supervision (auxiliary
  losses at each decoder scale). We omit it for simplicity in the first working
  version; noted as a limitation / possible future addition in the report.
- **Single shared `AttentionBlock2D`**: rather than the reference repo's
  per-experiment config-driven class selection, we use one configurable module
  (`attention_type=None/"spatial"/"cbam"/"hybrid"`) so all four model variants in
  our comparison come from one code path, reducing the chance of implementation
  drift between variants (relevant to the "clarity and structure of code" grading
  criterion).
- **Un-gated shallowest skip**: matches the paper's own stated design choice
  ("low-level feature-maps, i.e. the first skip connections, are not used in the
  gating function") — implemented in `models.py` by simply not wrapping `skip1`
  in an `AttentionBlock2D`.

## Our extension vs. the reference repo

Neither `grid_attention_layer.py` nor the paper implement any channel-wise
gating — the reference repo's attention is spatial-only, confirming that our
`ContextGatedChannelGate` (channel attention conditioned on both `x` and `g`,
see `src/layers.py`) is a genuine addition, not a re-derivation of something
already in the official code.
