"""
data.py
-------
Data acquisition and preprocessing for the Medical Segmentation Decathlon
Task09_Spleen dataset (41 labeled 3D abdominal CT volumes, binary spleen mask).

Pipeline:
  1. download_spleen_dataset() - downloads + extracts the official Decathlon
     Task09_Spleen archive via MONAI's DecathlonDataset helper (handles the
     dataset.json manifest, source URL, and checksum verification for us).
  2. build_2d_slice_dataset() - loads each 3D NIfTI volume/label pair, applies a
     soft-tissue HU window, normalizes, extracts axial slices, resizes to a fixed
     size, and writes them out as (image, mask) .npy pairs on disk so that
     training reads cheap 2D arrays instead of re-slicing 3D volumes every epoch.
  3. SpleenSliceDataset - a torch Dataset over the cached 2D slices, with a
     patient-level train/val/test split (never split at the slice level, to avoid
     leaking neighbouring slices of the same patient across splits).

Run this module's `main()` once (locally or in the first Colab cell) to produce
the cached 2D slice arrays; afterwards training just reads from disk.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - only needed when actually running preprocessing
    nib = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


# Soft-tissue HU window used for CT normalization (paper uses per-scan z-score after
# clipping; we follow standard abdominal soft-tissue windowing, which is a common,
# well-validated choice for spleen/organ segmentation on CT).
HU_MIN, HU_MAX = -125, 275


def download_spleen_dataset(root_dir: str) -> Path:
    """
    Downloads and extracts Task09_Spleen from the Medical Segmentation Decathlon
    using MONAI's built-in downloader (handles the hosted archive URL + checksum).

    Requires: `pip install monai nibabel`.

    Returns the path to the extracted "Task09_Spleen" folder, which contains
    imagesTr/ (39 labeled training volumes... note: Decathlon calls this split
    "training" even though we will re-split it ourselves below), labelsTr/, and
    a dataset.json manifest.
    """
    from monai.apps import DecathlonDataset

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # download=True fetches+extracts the archive on first call; subsequent calls
    # are no-ops if the expected files already exist. We only use this for the
    # download side-effect - actual loading happens in build_2d_slice_dataset().
    DecathlonDataset(root_dir=str(root_dir), task="Task09_Spleen", section="training",
                      transform=None, download=True, cache_num=0)

    return root_dir / "Task09_Spleen"


def _list_volume_label_pairs(task_dir: Path) -> List[Tuple[Path, Path]]:
    """Reads dataset.json to get matched (image, label) NIfTI file paths."""
    with open(task_dir / "dataset.json") as f:
        manifest = json.load(f)
    pairs = []
    for entry in manifest["training"]:
        img_path = task_dir / entry["image"].lstrip("./")
        lbl_path = task_dir / entry["label"].lstrip("./")
        pairs.append((img_path, lbl_path))
    return pairs


def _window_and_normalize(volume: np.ndarray) -> np.ndarray:
    """Clip to soft-tissue HU window, then rescale to [0, 1]."""
    volume = np.clip(volume, HU_MIN, HU_MAX)
    return (volume - HU_MIN) / (HU_MAX - HU_MIN)


def _resize_slice(slice_2d: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    """Resize a single 2D slice to (size, size). Nearest-neighbour for masks,
    bilinear for images, to avoid introducing fractional/soft labels."""
    from skimage.transform import resize
    order = 0 if is_mask else 1
    out = resize(slice_2d, (size, size), order=order, preserve_range=True, anti_aliasing=not is_mask)
    return out.astype(np.float32)


def build_2d_slice_dataset(task_dir: Path, out_dir: Path, image_size: int = 128,
                            min_spleen_pixels: int = 1, background_slice_ratio: float = 0.3,
                            seed: int = 42) -> None:
    """
    Converts the 3D volumes in `task_dir` into cached 2D axial slices in `out_dir`,
    organized as:
        out_dir/<patient_id>/img_<slice_idx>.npy
        out_dir/<patient_id>/msk_<slice_idx>.npy

    For each patient, keeps every slice containing at least `min_spleen_pixels`
    foreground pixels, plus a random sample of background-only slices (ratio
    controlled by `background_slice_ratio`) so the dataset isn't purely
    foreground-containing slices (which would make the task unrealistically easy
    and hide false-positive behaviour on empty slices).
    """
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _list_volume_label_pairs(task_dir)

    for img_path, lbl_path in pairs:
        patient_id = img_path.stem.replace(".nii", "")
        patient_dir = out_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        img_vol = nib.load(str(img_path)).get_fdata()
        lbl_vol = nib.load(str(lbl_path)).get_fdata()
        img_vol = _window_and_normalize(img_vol)

        fg_slices, bg_slices = [], []
        for z in range(img_vol.shape[2]):
            if lbl_vol[:, :, z].sum() >= min_spleen_pixels:
                fg_slices.append(z)
            else:
                bg_slices.append(z)

        n_bg_keep = int(len(fg_slices) * background_slice_ratio)
        keep_bg = rng.sample(bg_slices, min(n_bg_keep, len(bg_slices)))
        keep_slices = sorted(fg_slices + keep_bg)

        for z in keep_slices:
            img_2d = _resize_slice(img_vol[:, :, z], image_size, is_mask=False)
            msk_2d = _resize_slice((lbl_vol[:, :, z] > 0).astype(np.float32), image_size, is_mask=True)
            np.save(patient_dir / f"img_{z:04d}.npy", img_2d)
            np.save(patient_dir / f"msk_{z:04d}.npy", msk_2d)

        print(f"{patient_id}: kept {len(keep_slices)} slices "
              f"({len(fg_slices)} foreground, {len(keep_bg)} background)")


def patient_level_split(out_dir: Path, val_frac: float = 0.15, test_frac: float = 0.2,
                         seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Splits patient IDs (not slices!) into train/val/test to prevent leakage
    between slices of the same patient across splits."""
    patient_ids = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_ids = patient_ids[:n_test]
    val_ids = patient_ids[n_test:n_test + n_val]
    train_ids = patient_ids[n_test + n_val:]
    return train_ids, val_ids, test_ids


class SpleenSliceDataset(Dataset):
    """torch Dataset over cached 2D slices for a given list of patient IDs."""

    def __init__(self, slice_dir: Path, patient_ids: List[str], augment: bool = False):
        self.samples = []
        for pid in patient_ids:
            pdir = Path(slice_dir) / pid
            for img_file in sorted(pdir.glob("img_*.npy")):
                msk_file = pdir / img_file.name.replace("img_", "msk_")
                self.samples.append((img_file, msk_file))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_file, msk_file = self.samples[idx]
        img = np.load(img_file)[None, ...]  # (1, H, W)
        msk = np.load(msk_file)[None, ...]  # (1, H, W)

        if self.augment:
            img, msk = _augment(img, msk)

        return torch.from_numpy(img.copy()).float(), torch.from_numpy(msk.copy()).float()


def _augment(img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Light augmentation: random horizontal/vertical flip + 90-degree rotation.
    Kept simple/cheap for a Colab proof-of-concept; extend with elastic/affine
    transforms (as in the original paper) if training time allows."""
    if random.random() < 0.5:
        img, msk = img[:, :, ::-1], msk[:, :, ::-1]
    if random.random() < 0.5:
        img, msk = img[:, ::-1, :], msk[:, ::-1, :]
    k = random.choice([0, 1, 2, 3])
    if k:
        img, msk = np.rot90(img, k, axes=(1, 2)), np.rot90(msk, k, axes=(1, 2))
    return img, msk


def main():
    """Standalone entry point: download + preprocess. Intended to be run once,
    either locally or as an early Colab cell, before training starts."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./decathlon_data")
    parser.add_argument("--out", default="./slices_2d")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    task_dir = download_spleen_dataset(args.root)
    build_2d_slice_dataset(task_dir, Path(args.out), image_size=args.image_size)


if __name__ == "__main__":
    main()
