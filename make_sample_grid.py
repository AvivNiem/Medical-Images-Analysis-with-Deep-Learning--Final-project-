"""
make_sample_grid.py
-------------------
Generate a comparison grid: several test slices (rows) x all four model variants
(columns), each panel showing the predicted spleen (red) against the ground-truth
contour (blue), with the per-slice Dice printed. Uses the saved seed-0 weights.

Run on the server after run_all.py:
    CUDA_VISIBLE_DEVICES=1 python make_sample_grid.py --image-size 256 --n 5
Produces results/fig_sample_grid.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import patient_level_split, SpleenSliceDataset
from src.train import TrainConfig, get_device, load_trained_model
from src.metrics import dice_score

VARIANTS = [None, "spatial", "cbam", "hybrid"]
NAMES = {"None": "U-Net", "spatial": "Attention U-Net", "cbam": "AG + CBAM",
         "hybrid": "AG + context-gated (ours)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n", type=int, default=5, help="number of sample slices")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default="./results/results.json")
    args = ap.parse_args()

    device = get_device()
    results = json.load(open(args.results))
    slice_dir = Path(f"./slices_2d_{args.image_size}")
    cfg = TrainConfig(slice_dir=str(slice_dir), image_size=args.image_size)

    _, _, test_ids = patient_level_split(slice_dir, val_frac=cfg.val_frac,
                                         test_frac=cfg.test_frac, seed=1234)
    ds = SpleenSliceDataset(slice_dir, test_ids, augment=False)
    fg = [i for i in range(len(ds)) if ds[i][1].sum() > 0]
    # spread the sample slices across the foreground range (small, medium, large spleen)
    picks = [fg[int(k)] for k in np.linspace(len(fg)*0.1, len(fg)*0.9, args.n)]

    models = {str(a): load_trained_model(results, a, args.seed, cfg, device) for a in VARIANTS}

    ncol = len(VARIANTS) + 1
    fig, axes = plt.subplots(args.n, ncol, figsize=(2.6*ncol, 2.6*args.n))
    for r, idx in enumerate(picks):
        img, msk = ds[idx]
        img_np, msk_np = img.squeeze().numpy(), msk.squeeze().numpy()
        ax = axes[r, 0]
        ax.imshow(img_np, cmap="gray"); ax.contour(msk_np, [0.5], colors="deepskyblue", linewidths=1)
        ax.set_ylabel(f"slice {idx}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title("Input + GT", fontsize=10)
        for c, a in enumerate(VARIANTS, start=1):
            m = models[str(a)]
            with torch.no_grad():
                logits = m(img.unsqueeze(0).to(device))
                d = dice_score(logits, msk.unsqueeze(0).to(device)).item()
                pred = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
            ax = axes[r, c]
            ax.imshow(img_np, cmap="gray")
            ax.imshow(np.ma.masked_where(pred < 0.5, pred), cmap="autumn", alpha=0.5)
            ax.contour(msk_np, [0.5], colors="deepskyblue", linewidths=1)
            ax.set_title(f"{NAMES[str(a)]}\nDice={d:.2f}" if r == 0 else f"Dice={d:.2f}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Model predictions (red) vs. ground truth (blue) across sample slices", y=1.005)
    fig.tight_layout()
    out = "results/fig_sample_grid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
