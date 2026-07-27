"""
diag3d.py
---------
Diagnostic: train MONAI's *known-good* stock 3D U-Net through OUR data and
validation pipeline (fold 0), to isolate whether the poor convergence is caused by
our model (`model3d.py`) or by our data/validation pipeline (`data3d.py` +
`volume_dice`). MONAI's UNet reaches ~0.95 on this exact spleen task, so:

  * if this reaches ~0.9  -> the bug is in OUR model  -> fix model3d.py
  * if this also caps low -> the bug is in OUR data/val pipeline

Run:  CUDA_VISIBLE_DEVICES=1 python -m spleen_3d.diag3d
"""
import numpy as np
import torch

from .data3d import get_data_dicts, kfold_split, make_loaders, PATCH
from .train3d import get_device


def main():
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss
    from monai.inferers import sliding_window_inference

    device = get_device()
    print(f"Device: {device}")

    dicts = get_data_dicts("./decathlon_data/Task09_Spleen")
    train_dicts, val_dicts = kfold_split(dicts, k=5, fold=0)
    train_loader, val_loader = make_loaders(train_dicts, val_dicts, batch_size=2, num_workers=4)

    # MONAI's reference spleen U-Net (with residual units, the key difference).
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
    loss_fn = DiceLoss(sigmoid=True)

    @torch.no_grad()
    def val_dice():
        model.eval(); ds = []
        for b in val_loader:
            img = b["image"].to(device); lbl = (b["label"] > 0).float().to(device)
            logits = sliding_window_inference(img, PATCH, 2, model, overlap=0.25)
            pred = (torch.sigmoid(logits) > 0.5).float()
            tp = (pred * lbl).sum().item(); fp = (pred * (1 - lbl)).sum().item()
            fn = ((1 - pred) * lbl).sum().item()
            ds.append((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6))
        return float(np.mean(ds))

    best = 0.0
    for epoch in range(120):
        model.train()
        for b in train_loader:
            img = b["image"].to(device); lbl = (b["label"] > 0).float().to(device)
            opt.zero_grad()
            loss = loss_fn(model(img), lbl)
            loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 3 == 0:
            vd = val_dice(); best = max(best, vd)
            print(f"  [MONAI-UNet fold0] epoch {epoch+1:3d} val_dice={vd:.4f} best={best:.4f}")

    print(f"\nDIAG result: MONAI stock UNet best Dice {best:.4f}")
    print("If ~0.9 -> our model3d.py is the problem. If capped low -> our data/val pipeline is.")


if __name__ == "__main__":
    main()
