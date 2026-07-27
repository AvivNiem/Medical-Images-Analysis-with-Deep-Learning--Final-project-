"""
faithful_2d/model.py
--------------------
2D Attention U-Net WITH deep supervision — the faithful reproduction adapted to 2D
data. Reuses the proven 2D gate blocks from `src/layers.py` (the grid attention
gate is a direct port of the paper's implementation), and adds auxiliary
segmentation heads at each decoder scale (deep supervision), as in the paper's
`*_dsv` configuration.

Only `attention_type` differs between the four compared variants:
  None -> U-Net, "spatial" -> Attention U-Net, "cbam" -> AG+CBAM, "hybrid" -> ours.

Output: main logits (B, num_classes, H, W) plus a list of deep-supervision logits
at the same resolution (empty if deep_supervision=False).
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import ConvBlock, DownBlock, UpBlock, AttentionBlock2D


class FaithfulAttentionUNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_channels: int = 32,
                 attention_type: Optional[str] = None, deep_supervision: bool = True):
        super().__init__()
        assert attention_type in (None, "spatial", "cbam", "hybrid")
        self.attention_type = attention_type
        self.deep_supervision = deep_supervision
        c = base_channels

        self.down1 = DownBlock(in_channels, c)
        self.down2 = DownBlock(c, c * 2)
        self.down3 = DownBlock(c * 2, c * 4)
        self.down4 = DownBlock(c * 4, c * 8)
        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.attn4 = AttentionBlock2D(c * 8, c * 16, attention_type)
        self.attn3 = AttentionBlock2D(c * 4, c * 8, attention_type)
        self.attn2 = AttentionBlock2D(c * 2, c * 4, attention_type)

        self.up4 = UpBlock(c * 16, c * 8, c * 8)
        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)

        self.out_conv = nn.Conv2d(c, num_classes, 1)
        if deep_supervision:
            self.dsv4 = nn.Conv2d(c * 8, num_classes, 1)
            self.dsv3 = nn.Conv2d(c * 4, num_classes, 1)
            self.dsv2 = nn.Conv2d(c * 2, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        d3, skip3 = self.down3(d2)
        d4, skip4 = self.down4(d3)
        b = self.bottleneck(d4)

        u4 = self.up4(b, self.attn4(skip4, b))
        u3 = self.up3(u4, self.attn3(skip3, u4))
        u2 = self.up2(u3, self.attn2(skip2, u3))
        u1 = self.up1(u2, skip1)

        main = self.out_conv(u1)
        aux: List[torch.Tensor] = []
        if self.deep_supervision:
            for head, feat in [(self.dsv4, u4), (self.dsv3, u3), (self.dsv2, u2)]:
                aux.append(F.interpolate(head(feat), size=main.shape[-2:],
                                         mode="bilinear", align_corners=False))
        return main, aux

    def get_attention_maps(self):
        return {n: {"alpha": b.last_alpha, "beta": b.last_beta}
                for n, b in [("level4", self.attn4), ("level3", self.attn3), ("level2", self.attn2)]}


def build_faithful_model(attention_type, base_channels=32, deep_supervision=True):
    return FaithfulAttentionUNet2D(attention_type=attention_type, base_channels=base_channels,
                                   deep_supervision=deep_supervision)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
