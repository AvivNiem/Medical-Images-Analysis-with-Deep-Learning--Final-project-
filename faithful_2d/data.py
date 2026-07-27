"""
faithful_2d/data.py
-------------------
Data for the faithful 2D reproduction. Reuses the cached 2D slices from the main
pipeline (the file listing logic of `SpleenSliceDataset`) but adds two paper-style
elements:

  * per-slice z-score normalization (zero mean, unit std) -> closer to the paper's
    N(0,1) intensity normalization than the plain [0,1] window used on main.
  * affine augmentation (small rotation + scale) plus flips -> the paper uses
    "affine transformations, axial flips" rather than our earlier flips + 90-deg.

Image and mask receive the SAME geometric transform (mask with nearest-neighbour /
order 0) so they stay aligned.
"""
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.ndimage import rotate as nd_rotate, zoom as nd_zoom
except ImportError:  # pragma: no cover
    nd_rotate = nd_zoom = None


def _list_samples(slice_dir: str, patient_ids: List[str]):
    samples = []
    for pid in patient_ids:
        pdir = Path(slice_dir) / pid
        for img_file in sorted(pdir.glob("img_*.npy")):
            samples.append((img_file, pdir / img_file.name.replace("img_", "msk_")))
    return samples


class FaithfulSpleenDataset(Dataset):
    """2D slice dataset with z-score normalization and affine+flip augmentation."""

    def __init__(self, slice_dir: str, patient_ids: List[str], augment: bool = False,
                 zscore: bool = True):
        self.samples = _list_samples(slice_dir, patient_ids)
        self.augment = augment
        self.zscore = zscore

    def __len__(self):
        return len(self.samples)

    def _affine(self, img, msk):
        # flips
        if random.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if random.random() < 0.5:
            img, msk = img[::-1, :], msk[::-1, :]
        # small rotation (paper-style affine)
        if nd_rotate is not None and random.random() < 0.5:
            ang = random.uniform(-15, 15)
            img = nd_rotate(img, ang, reshape=False, order=1, mode="nearest")
            msk = nd_rotate(msk, ang, reshape=False, order=0, mode="nearest")
        # small scale (affine)
        if nd_zoom is not None and random.random() < 0.3:
            s = random.uniform(0.9, 1.1)
            h, w = img.shape
            zi = nd_zoom(img, s, order=1)
            zm = nd_zoom(msk, s, order=0)
            img, msk = _center_crop_or_pad(zi, (h, w)), _center_crop_or_pad(zm, (h, w))
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

    def __getitem__(self, idx):
        img_file, msk_file = self.samples[idx]
        img = np.load(img_file).astype(np.float32)
        msk = (np.load(msk_file) > 0).astype(np.float32)

        if self.zscore:
            img = (img - img.mean()) / (img.std() + 1e-6)
        if self.augment:
            img, msk = self._affine(img, msk)

        return (torch.from_numpy(img[None].copy()).float(),
                torch.from_numpy(msk[None].copy()).float())


def _center_crop_or_pad(a, target):
    """Crop or pad a 2D array to the target (H, W) around the center."""
    th, tw = target
    h, w = a.shape
    out = np.zeros(target, dtype=a.dtype)
    # source crop box
    sy0 = max((h - th) // 2, 0); sx0 = max((w - tw) // 2, 0)
    sy1 = sy0 + min(th, h); sx1 = sx0 + min(tw, w)
    # dest paste box
    dy0 = max((th - h) // 2, 0); dx0 = max((tw - w) // 2, 0)
    crop = a[sy0:sy1, sx0:sx1]
    out[dy0:dy0 + crop.shape[0], dx0:dx0 + crop.shape[1]] = crop
    return out


def list_patients(slice_dir: str) -> List[str]:
    return sorted(p.name for p in Path(slice_dir).iterdir() if p.is_dir())
