"""
visualize.py
------------
Qualitative visualizations for the report's Results & Analysis section:

  1. plot_training_curves()      - train loss & val Dice per epoch, per variant.
  2. plot_prediction_overlay()   - image / GT / prediction comparison across
                                   variants on the same slices (like paper Fig. 3b).
  3. plot_spatial_attention()    - the spatial attention map alpha overlaid on the
                                   input, per gated level (like paper Fig. 3a/4).
  4. plot_channel_attention()    - OUR extension: the per-channel weights beta from
                                   the context-gated channel gate, to inspect
                                   whether channel attention learns a meaningful,
                                   non-uniform selection (a key piece of evidence
                                   that the extension does something the spatial
                                   gate cannot).
  5. show_worst_cases()          - the lowest-Dice test slices, for failure analysis.

All functions take already-trained models / cached tensors and use matplotlib only,
so they run unchanged on Colab.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

import matplotlib.pyplot as plt


VARIANT_LABELS = {
    "None": "U-Net",
    "spatial": "Attention U-Net",
    "cbam": "AG + CBAM",
    "hybrid": "AG + context-gated (ours)",
}


def plot_training_curves(results: Dict, save_path: Optional[str] = None):
    """Overlay val-Dice curves (mean across seeds) for all variants."""
    from collections import defaultdict
    curves = defaultdict(list)
    for run in results["runs"]:
        curves[run["attention_type"]].append(run["history"]["val_dice"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for att, seed_curves in curves.items():
        # pad ragged (early-stopped) curves to equal length with their last value
        max_len = max(len(c) for c in seed_curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in seed_curves])
        mean, std = padded.mean(0), padded.std(0)
        epochs = np.arange(max_len)
        ax.plot(epochs, mean, label=VARIANT_LABELS.get(att, att))
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.set_title("Validation Dice across training (mean ± std over seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def plot_prediction_overlay(models: Dict[str, "torch.nn.Module"], img: "torch.Tensor",
                            mask: "torch.Tensor", device, save_path: Optional[str] = None):
    """Compare predictions from all variants on a single slice.

    Layout: [input+GT contour] followed by one column per model showing its
    predicted mask (green) against the GT contour (blue), mirroring paper Fig. 3b."""
    img_np = img.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()

    n = len(models) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))

    axes[0].imshow(img_np, cmap="gray")
    axes[0].contour(mask_np, levels=[0.5], colors="deepskyblue", linewidths=1.2)
    axes[0].set_title("Input + GT")
    axes[0].axis("off")

    for i, (att, model) in enumerate(models.items(), start=1):
        model.eval()
        logits = model(img.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
        axes[i].imshow(img_np, cmap="gray")
        axes[i].imshow(np.ma.masked_where(pred < 0.5, pred), cmap="autumn", alpha=0.5)
        axes[i].contour(mask_np, levels=[0.5], colors="deepskyblue", linewidths=1.0)
        axes[i].set_title(VARIANT_LABELS.get(att, att), fontsize=9)
        axes[i].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def plot_spatial_attention(model, img: "torch.Tensor", device,
                           save_path: Optional[str] = None):
    """Overlay the spatial attention map alpha (upsampled to input resolution) for
    each gated level, reproducing the style of paper Fig. 3a / Fig. 4."""
    model.eval()
    _ = model(img.unsqueeze(0).to(device))
    maps = model.get_attention_maps()
    img_np = img.squeeze().cpu().numpy()

    levels = [(k, v["alpha"]) for k, v in maps.items() if v["alpha"] is not None]
    n = len(levels) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    for i, (level, alpha) in enumerate(levels, start=1):
        a = alpha.squeeze().cpu().numpy()
        # resize to input size for overlay
        from skimage.transform import resize
        a = resize(a, img_np.shape, order=1, preserve_range=True)
        axes[i].imshow(img_np, cmap="gray")
        axes[i].imshow(a, cmap="jet", alpha=0.5)
        axes[i].set_title(f"alpha @ {level}", fontsize=9)
        axes[i].axis("off")

    fig.suptitle("Spatial attention coefficients", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def plot_channel_attention(model, img: "torch.Tensor", device,
                           save_path: Optional[str] = None):
    """Bar plot of the context-gated channel weights beta per gated level (OUR
    extension). A near-uniform beta would mean the channel gate learned nothing;
    a spread-out beta is evidence it performs meaningful channel selection."""
    model.eval()
    _ = model(img.unsqueeze(0).to(device))
    maps = model.get_attention_maps()

    levels = [(k, v["beta"]) for k, v in maps.items() if v["beta"] is not None]
    if not levels:
        print("No channel attention maps found (model is not the 'hybrid' variant).")
        return None

    fig, axes = plt.subplots(len(levels), 1, figsize=(7, 2.2 * len(levels)))
    if len(levels) == 1:
        axes = [axes]

    for ax, (level, beta) in zip(axes, levels):
        b = beta.squeeze().cpu().numpy()
        ax.bar(np.arange(len(b)), np.sort(b)[::-1])
        ax.axhline(b.mean(), color="red", ls="--", lw=1,
                   label=f"mean={b.mean():.2f}, std={b.std():.2f}")
        ax.set_title(f"channel weights beta @ {level} (sorted)", fontsize=9)
        ax.set_xlabel("channel (sorted by weight)")
        ax.set_ylabel("beta")
        ax.legend(fontsize=8)

    fig.suptitle("Context-gated channel attention (ours)", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def show_worst_cases(model, dataset, device, k: int = 4, save_path: Optional[str] = None):
    """Find and display the k lowest-Dice foreground slices for failure analysis."""
    from .metrics import dice_score
    model.eval()

    scored = []
    for idx in range(len(dataset)):
        img, msk = dataset[idx]
        if msk.sum() == 0:
            continue
        logits = model(img.unsqueeze(0).to(device))
        d = dice_score(logits, msk.unsqueeze(0).to(device)).item()
        scored.append((d, idx))

    scored.sort()
    worst = scored[:k]

    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.2))
    if k == 1:
        axes = [axes]
    for ax, (d, idx) in zip(axes, worst):
        img, msk = dataset[idx]
        logits = model(img.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
        ax.imshow(img.squeeze().cpu().numpy(), cmap="gray")
        ax.imshow(np.ma.masked_where(pred < 0.5, pred), cmap="autumn", alpha=0.5)
        ax.contour(msk.squeeze().cpu().numpy(), levels=[0.5], colors="deepskyblue", linewidths=1.0)
        ax.set_title(f"Dice={d:.2f}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Worst-case predictions (failure analysis)", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
