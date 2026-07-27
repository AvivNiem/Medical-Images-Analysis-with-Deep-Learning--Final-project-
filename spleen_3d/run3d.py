"""
run3d.py
--------
Entry point for the 3D faithful reproduction. Runs a forward-pass shape check for
all four variants, then k-fold cross-validation.

    # 5-minute smoke test (1 fold, few epochs, tiny net) — verifies end-to-end:
    CUDA_VISIBLE_DEVICES=1 python -m spleen_3d.run3d --quick

    # Full reported run (5-fold CV, all variants):
    CUDA_VISIBLE_DEVICES=1 python -m spleen_3d.run3d

Run from the repository root so the `spleen_3d` package imports resolve.
"""
import argparse
import torch

from .model3d import build_model3d, count_parameters
from .train3d import Cfg3D, run_kfold, get_device


def shape_check(base=16):
    device = get_device()
    x = torch.randn(1, 1, 96, 96, 96, device=device)
    print(f"Shape check on {device} (input {tuple(x.shape)}):")
    for att in [None, "spatial", "cbam", "hybrid"]:
        m = build_model3d(att, base=base).to(device)
        main, aux = m(x)
        assert main.shape == (1, 1, 96, 96, 96), (att, main.shape)
        print(f"  {str(att):8s} out={tuple(main.shape)} aux={len(aux)} params={count_parameters(m):,}")
    print("All 3D variants build and produce correct output shapes.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default="./decathlon_data/Task09_Spleen")
    ap.add_argument("--out-dir", default="./results_3d")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    args = ap.parse_args()

    if args.quick:
        # Tiny end-to-end check: fold 0 of two variants, small net, few epochs.
        from pathlib import Path
        from .data3d import get_data_dicts
        from .train3d import train_one_fold

        shape_check(base=8)
        cfg = Cfg3D(task_dir=args.task_dir, out_dir=args.out_dir,
                    base=8, max_epochs=6, val_interval=3, patience=3,
                    attention_types=[None, "hybrid"])
        device = get_device()
        weights_dir = Path(cfg.out_dir) / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        dicts = get_data_dicts(cfg.task_dir)
        for att in cfg.attention_types:
            r = train_one_fold(att, 0, dicts, cfg, device, weights_dir)
            print(f"  SMOKE {att}: fold0 Dice {r['val_dice']:.4f}")
        print("\nSmoke test complete — pipeline runs end-to-end.")
    else:
        shape_check(base=16)
        cfg = Cfg3D(task_dir=args.task_dir, out_dir=args.out_dir)
        run_kfold(cfg)


if __name__ == "__main__":
    main()
