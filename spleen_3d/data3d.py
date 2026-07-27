"""
data3d.py
---------
3D data pipeline for the faithful Attention U-Net reproduction on Decathlon
Task09_Spleen. Uses MONAI transforms for robust 3D medical-image handling
(loading, spacing, intensity windowing, patch sampling, augmentation) — the model
itself is our own faithful reproduction; MONAI only handles data plumbing.

Key design choices (to match the paper's 3D setup):
  * Volumes resampled to isotropic 1.5 mm, oriented RAS, windowed to a soft-tissue
    HU range and scaled to [0, 1], cropped to the foreground.
  * Training uses PATCH sampling (RandCropByPosNegLabeld) — random 96^3 sub-volumes
    with a balanced ratio of foreground/background patches, which addresses the
    class imbalance that forced a focal loss in the 2D proof-of-concept (so here we
    can use the paper's plain Dice loss).
  * Augmentation: random affine (rotation/scale) + axis flips — the paper's
    "affine transformations, axial flips".
  * Validation/test: full volumes, evaluated by sliding-window inference (see infer3d).
  * Patient-level k-fold cross-validation splitter.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# HU soft-tissue window (same rationale as the 2D pipeline)
HU_MIN, HU_MAX = -125, 275
PATCH = (96, 96, 96)
SPACING = (1.5, 1.5, 1.5)


def get_data_dicts(task_dir) -> List[Dict[str, str]]:
    """Read dataset.json → list of {'image': path, 'label': path} for the 41 labeled
    Task09_Spleen volumes."""
    task_dir = Path(task_dir)
    manifest = json.load(open(task_dir / "dataset.json"))
    dicts = []
    for e in manifest["training"]:
        dicts.append({"image": str(task_dir / e["image"].lstrip("./")),
                      "label": str(task_dir / e["label"].lstrip("./"))})
    return dicts


def kfold_split(dicts: List[Dict], k: int, fold: int, seed: int = 1234
                ) -> Tuple[List[Dict], List[Dict]]:
    """Patient-level k-fold split. Returns (train_dicts, val_dicts) for the given
    fold index (0..k-1). Deterministic given the seed, so every model variant sees
    the identical folds."""
    idx = np.arange(len(dicts))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    val_idx = folds[fold]
    train_idx = np.concatenate([folds[i] for i in range(k) if i != fold])
    return [dicts[i] for i in train_idx], [dicts[i] for i in val_idx]


def train_transforms(patch=PATCH, spacing=SPACING):
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
                                  Spacingd, ScaleIntensityRanged, CropForegroundd,
                                  RandCropByPosNegLabeld, RandAffined, RandFlipd, ToTensord)
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Patch sampling: 4 patches/volume, half centred on spleen (pos), half background (neg).
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=patch,
                               pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
        # Paper-style augmentation: affine (rotation/scale) + axial flips.
        RandAffined(keys=["image", "label"], prob=0.3, rotate_range=(0.26, 0.26, 0.26),
                    scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ToTensord(keys=["image", "label"]),
    ])


def val_transforms(spacing=SPACING):
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
                                  Spacingd, ScaleIntensityRanged, CropForegroundd, ToTensord)
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])


def make_loaders(train_dicts, val_dicts, batch_size=2, cache=True, num_workers=4):
    """Build MONAI train/val DataLoaders. CacheDataset speeds up repeated epochs by
    caching the deterministic preprocessing."""
    from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate
    tr_tf, va_tf = train_transforms(), val_transforms()
    DS = CacheDataset if cache else Dataset
    train_ds = DS(data=train_dicts, transform=tr_tf, cache_rate=1.0 if cache else 0.0)
    val_ds = DS(data=val_dicts, transform=va_tf, cache_rate=1.0 if cache else 0.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
