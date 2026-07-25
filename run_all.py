"""
run_all.py
----------
Standalone driver to run the full experiment headlessly on a GPU server (no
notebook needed). Mirrors the Colab notebook exactly, so results are identical.

It (1) downloads + slices the data, (2) runs the 4-way main comparison, (3) runs
the low-data-regime sweep, and (4) saves the results table, significance tests,
and all figures into the results directory.

Usage
-----
  # Full configuration used for the reported results (e.g. on an L40):
  python run_all.py --full

  # Quick proof-of-concept (same as the Colab QUICK default):
  python run_all.py

  # Custom:
  python run_all.py --image-size 256 --seeds 0 1 2 3 4 --epochs 60 \
                    --sweep-sizes 4 8 16 all
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures to files, no display needed
import matplotlib.pyplot as plt

from src.data import (download_spleen_dataset, build_2d_slice_dataset,
                       patient_level_split, SpleenSliceDataset)
from src.train import (TrainConfig, run_experiment, run_low_data_sweep,
                       load_trained_model, get_device)
from src.stats import (aggregate_by_variant, format_results_table,
                       aggregate_sweep, paired_wilcoxon)
from src.visualize import (plot_training_curves, plot_prediction_overlay,
                           plot_spatial_attention, plot_channel_attention,
                           show_worst_cases, plot_low_data_curve)
from src.metrics import dice_score
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true",
                   help="Use the full reported config (256px, 5 seeds, 4 sweep sizes).")
    p.add_argument("--data-root", default="./decathlon_data")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--sweep-sizes", nargs="+", default=None,
                   help="Training sizes for the sweep; use 'all' for the full set.")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def resolve_config(args):
    """Merge CLI overrides with the QUICK/FULL presets."""
    if args.full:
        seeds, epochs, img, sweep = [0, 1, 2, 3, 4], 60, 256, [4, 8, 16, None]
    else:
        seeds, epochs, img, sweep = [0, 1], 30, 128, [8, None]

    if args.seeds is not None:
        seeds = args.seeds
    if args.epochs is not None:
        epochs = args.epochs
    if args.image_size is not None:
        img = args.image_size
    if args.sweep_sizes is not None:
        sweep = [None if s == "all" else int(s) for s in args.sweep_sizes]
    return seeds, epochs, img, sweep


def main():
    args = parse_args()
    seeds, epochs, img, sweep_sizes = resolve_config(args)
    device = get_device()
    print(f"Device: {device} | image_size={img} | seeds={seeds} | epochs={epochs} "
          f"| sweep_sizes={sweep_sizes}")

    # 1) Data ---------------------------------------------------------------
    slice_dir = Path(f"./slices_2d_{img}")
    if not slice_dir.exists() or not any(slice_dir.iterdir()):
        task_dir = download_spleen_dataset(args.data_root)
        build_2d_slice_dataset(task_dir, slice_dir, image_size=img)
    else:
        print(f"Using cached slices at {slice_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        slice_dir=str(slice_dir),
        attention_types=[None, "spatial", "cbam", "hybrid"],
        seeds=seeds,
        max_epochs=epochs,
        batch_size=args.batch_size,
        lr=3e-4,
        grad_clip=1.0,
        loss_name="dice_focal",
        image_size=img,
        out_dir=str(out_dir),
    )

    # 2) Main 4-way comparison ---------------------------------------------
    print("\n########## MAIN COMPARISON ##########")
    results = run_experiment(cfg)
    summary = aggregate_by_variant(results)
    print("\n" + format_results_table(summary))

    # 3) Statistical significance on the shared test set --------------------
    train_ids, val_ids, test_ids = patient_level_split(
        slice_dir, val_frac=cfg.val_frac, test_frac=cfg.test_frac, seed=1234)
    test_ds = SpleenSliceDataset(slice_dir, test_ids, augment=False)
    fg_idx = [i for i in range(len(test_ds)) if test_ds[i][1].sum() > 0]

    def per_slice(model):
        model.eval(); out = []
        with torch.no_grad():
            for i in fg_idx:
                im, mk = test_ds[i]
                out.append(dice_score(model(im.unsqueeze(0).to(device)),
                                      mk.unsqueeze(0).to(device)).item())
        return out

    ps = {}
    for att in [None, "spatial", "cbam", "hybrid"]:
        ps[str(att)] = per_slice(load_trained_model(results, att, seeds[0], cfg, device))
    print("\nPaired Wilcoxon (hybrid vs baselines, per-slice Dice):")
    for base in ["None", "spatial", "cbam"]:
        t = paired_wilcoxon(ps, "hybrid", base)
        print(f"  hybrid vs {base:8s}: median diff {t['median_diff']:+.3f}  p={t['p_value']:.4f}")

    # 4) Qualitative figures ------------------------------------------------
    plot_training_curves(results, save_path=str(out_dir / "fig_training_curves.png"))
    models = {str(a): load_trained_model(results, a, seeds[0], cfg, device)
              for a in [None, "spatial", "cbam", "hybrid"]}
    img_t, msk_t = test_ds[fg_idx[len(fg_idx) // 2]]
    plot_prediction_overlay(models, img_t, msk_t, device, save_path=str(out_dir / "fig_overlay.png"))
    plot_spatial_attention(models["spatial"], img_t, device, save_path=str(out_dir / "fig_spatial_attn.png"))
    plot_channel_attention(models["hybrid"], img_t, device, save_path=str(out_dir / "fig_channel_attn.png"))
    show_worst_cases(models["hybrid"], test_ds, device, k=4, save_path=str(out_dir / "fig_worst_cases.png"))

    # 5) Low-data-regime sweep ---------------------------------------------
    print("\n########## LOW-DATA SWEEP ##########")
    sweep = run_low_data_sweep(cfg, train_sizes=sweep_sizes)
    sweep_agg = aggregate_sweep(sweep)
    plot_low_data_curve(sweep_agg, save_path=str(out_dir / "fig_low_data_curve.png"))
    print("\nLow-data sweep summary (test Dice, mean per variant/size):")
    for att, per in sweep_agg["summary"].items():
        row = "  ".join(f"n={k}:{v['dice_mean']:.3f}" for k, v in per.items())
        print(f"  {att:8s} {row}")

    print(f"\nAll results + figures saved under {out_dir}/")


if __name__ == "__main__":
    main()
