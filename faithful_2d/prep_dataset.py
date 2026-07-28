"""
faithful_2d/prep_dataset.py
--------------------------
Download and 2D-slice any Medical Segmentation Decathlon task so the faithful 2D
pipeline can run on it. Used to add the PANCREAS (Task07) — the paper's own, hard,
low-contrast organ — alongside the spleen.

    # ~11 GB download; run inside tmux. Subsamples to keep it a proof-of-concept.
    python -m faithful_2d.prep_dataset --task Task07_Pancreas \
           --out ./slices_2d_pancreas_256 --image-size 256 --max-patients 60

Produces the same on-disk 2D slice layout as the spleen data, so afterwards:
    python -m faithful_2d.train --slice-dir ./slices_2d_pancreas_256 --k 5 --epochs 80
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import nibabel as nib
from skimage.transform import resize


def download_task(root: str, task: str) -> Path:
    """Download+extract a Decathlon task via MONAI (no-op if already present)."""
    from monai.apps import DecathlonDataset
    Path(root).mkdir(parents=True, exist_ok=True)
    DecathlonDataset(root_dir=str(root), task=task, section="training",
                     transform=None, download=True, cache_num=0)
    return Path(root) / task


def slice_task(task_dir, out_dir, image_size=256, window=(-125, 275),
               max_patients=None, bg_ratio=0.3, seed=42):
    """Slice 3D volumes to 2D axial (image, mask) .npy pairs. Binary target =
    (label > 0) (for pancreas this is pancreas + tumour, the whole organ region).
    Keeps all foreground slices plus a sampled fraction of background slices."""
    manifest = json.load(open(Path(task_dir) / "dataset.json"))
    pairs = [(Path(task_dir) / e["image"].lstrip("./"),
              Path(task_dir) / e["label"].lstrip("./")) for e in manifest["training"]]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_patients:
        pairs = pairs[:max_patients]

    lo, hi = window
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in pairs:
        pid = img_path.stem.replace(".nii", "")
        pdir = out_dir / pid; pdir.mkdir(parents=True, exist_ok=True)

        img = nib.load(str(img_path)).get_fdata()
        lbl = nib.load(str(lbl_path)).get_fdata()
        img = np.clip(img, lo, hi); img = (img - lo) / (hi - lo)

        fg = [z for z in range(img.shape[2]) if (lbl[:, :, z] > 0).sum() > 0]
        bg = [z for z in range(img.shape[2]) if (lbl[:, :, z] > 0).sum() == 0]
        keep_bg = rng.sample(bg, min(int(len(fg) * bg_ratio), len(bg)))

        for z in sorted(fg + keep_bg):
            im = resize(img[:, :, z], (image_size, image_size), order=1,
                        preserve_range=True).astype(np.float32)
            mk = resize((lbl[:, :, z] > 0).astype(np.float32), (image_size, image_size),
                        order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
            np.save(pdir / f"img_{z:04d}.npy", im)
            np.save(pdir / f"msk_{z:04d}.npy", mk)
        print(f"{pid}: {len(fg)} fg + {len(keep_bg)} bg slices")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Task07_Pancreas")
    ap.add_argument("--root", default="./decathlon_data")
    ap.add_argument("--out", default="./slices_2d_pancreas_256")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--max-patients", type=int, default=60)
    ap.add_argument("--window", type=int, nargs=2, default=[-125, 275])
    args = ap.parse_args()

    task_dir = download_task(args.root, args.task)
    slice_task(task_dir, args.out, image_size=args.image_size,
               window=tuple(args.window), max_patients=args.max_patients)
    print(f"\nDone. Now run:\n  python -m faithful_2d.train --slice-dir {args.out} --k 5 --epochs 80")
