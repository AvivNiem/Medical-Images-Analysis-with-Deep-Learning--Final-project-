"""
model3d.py
----------
Configurable 3D Attention U-Net for the faithful reproduction. A single network
that becomes any of the four variants via `attention_type`, so the ONLY difference
between the compared models is the attention module on the skip connections:

    None      -> vanilla 3D U-Net (baseline)
    "spatial" -> Attention U-Net (paper's grid attention gate, reproduced)
    "cbam"    -> AG + CBAM (control)
    "hybrid"  -> AG + context-gated channel attention (ours)

Faithful-to-paper choices:
  * 3D convolutions on volumes/patches.
  * Deep supervision: auxiliary segmentation heads at each decoder scale, upsampled
    to full resolution and supervised jointly (as in the paper's *_dsv configs).
  * Shallowest skip left ungated (paper's stated design).

Input:  (B, 1, D, H, W).  Output: main logits (B, num_classes, D, H, W), plus a list
of deep-supervision logits at the same size (empty when deep_supervision=False).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers3d import ConvBlock3D, Down3D, Up3D, AttentionGate3D


class AttentionUNet3D(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 1, base: int = 16,
                 attention_type: Optional[str] = None, deep_supervision: bool = True):
        super().__init__()
        assert attention_type in (None, "spatial", "cbam", "hybrid")
        self.attention_type = attention_type
        self.deep_supervision = deep_supervision
        c = base

        # Encoder
        self.down1 = Down3D(in_ch, c)         # skip1: c    (ungated)
        self.down2 = Down3D(c, c * 2)          # skip2: 2c
        self.down3 = Down3D(c * 2, c * 4)       # skip3: 4c
        self.down4 = Down3D(c * 4, c * 8)       # skip4: 8c
        self.bottleneck = ConvBlock3D(c * 8, c * 16)

        # Attention gates (skip4, skip3, skip2)
        self.attn4 = AttentionGate3D(c * 8, c * 16, attention_type)
        self.attn3 = AttentionGate3D(c * 4, c * 8, attention_type)
        self.attn2 = AttentionGate3D(c * 2, c * 4, attention_type)

        # Decoder
        self.up4 = Up3D(c * 16, c * 8, c * 8)
        self.up3 = Up3D(c * 8, c * 4, c * 4)
        self.up2 = Up3D(c * 4, c * 2, c * 2)
        self.up1 = Up3D(c * 2, c, c)

        self.out_conv = nn.Conv3d(c, num_classes, 1)

        # Deep-supervision heads at the three deeper decoder scales
        if deep_supervision:
            self.dsv4 = nn.Conv3d(c * 8, num_classes, 1)
            self.dsv3 = nn.Conv3d(c * 4, num_classes, 1)
            self.dsv2 = nn.Conv3d(c * 2, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        d3, skip3 = self.down3(d2)
        d4, skip4 = self.down4(d3)
        b = self.bottleneck(d4)

        u4 = self.up4(b, self.attn4(skip4, b))
        u3 = self.up3(u4, self.attn3(skip3, u4))
        u2 = self.up2(u3, self.attn2(skip2, u3))
        u1 = self.up1(u2, skip1)  # shallowest skip ungated

        main = self.out_conv(u1)

        aux: List[torch.Tensor] = []
        if self.deep_supervision:
            for head, feat in [(self.dsv4, u4), (self.dsv3, u3), (self.dsv2, u2)]:
                aux.append(F.interpolate(head(feat), size=main.shape[2:],
                                         mode="trilinear", align_corners=False))
        return main, aux

    def get_attention_maps(self):
        return {name: {"alpha": blk.last_alpha, "beta": blk.last_beta}
                for name, blk in [("level4", self.attn4), ("level3", self.attn3),
                                  ("level2", self.attn2)]}


def build_model3d(attention_type: Optional[str], base: int = 16, num_classes: int = 1,
                  deep_supervision: bool = True) -> AttentionUNet3D:
    return AttentionUNet3D(in_ch=1, num_classes=num_classes, base=base,
                           attention_type=attention_type, deep_supervision=deep_supervision)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
