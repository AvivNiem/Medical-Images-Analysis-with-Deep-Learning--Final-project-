"""
models.py
---------
A single configurable 2D U-Net that can be instantiated as any of the four model
variants compared in this project, all sharing one encoder/decoder implementation
so the only difference between them is the `attention_type` argument:

    attention_type=None       -> vanilla U-Net                          (paper baseline)
    attention_type="spatial"  -> Attention U-Net                        (Oktay et al. 2018, reproduced)
    attention_type="cbam"     -> Attention U-Net + naive CBAM bolt-on   (control baseline,
                                  reproduces the ASCU-Net / CBAM-AG-UNet style of combination)
    attention_type="hybrid"   -> Attention U-Net + context-gated channel attention
                                  (OUR proposed extension)

Following the original paper's own design note, the shallowest (highest-resolution)
skip connection is NOT gated, since low-level features don't yet carry enough
semantic content for the gating signal to usefully condition on (see Section 2,
"Attention Gates in U-Net Model" in Oktay et al. 2018).

Input: single-channel 2D CT slices, shape (B, 1, H, W).
Output: single-channel logits, shape (B, 1, H, W) (binary spleen vs. background;
        pass through sigmoid outside the model, e.g. in the loss function).
"""

from typing import Optional

import torch
import torch.nn as nn

from .layers import ConvBlock, DownBlock, UpBlock, AttentionBlock2D


class AttentionUNet2D(nn.Module):
    """Configurable 2D U-Net. See module docstring for `attention_type` options."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        attention_type: Optional[str] = None,
    ):
        super().__init__()
        assert attention_type in (None, "spatial", "cbam", "hybrid")
        self.attention_type = attention_type
        c = base_channels

        # Encoder
        self.down1 = DownBlock(in_channels, c)       # skip1: c        @ H
        self.down2 = DownBlock(c, c * 2)              # skip2: 2c       @ H/2
        self.down3 = DownBlock(c * 2, c * 4)           # skip3: 4c       @ H/4
        self.down4 = DownBlock(c * 4, c * 8)           # skip4: 8c       @ H/8
        self.bottleneck = ConvBlock(c * 8, c * 16)     # b:     16c      @ H/16

        # Attention gates (skip4, skip3, skip2 only - skip1 stays ungated)
        self.attn4 = AttentionBlock2D(c * 8, c * 16, attention_type)
        self.attn3 = AttentionBlock2D(c * 4, c * 8, attention_type)
        self.attn2 = AttentionBlock2D(c * 2, c * 4, attention_type)

        # Decoder
        self.up4 = UpBlock(c * 16, c * 8, c * 8)
        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)

        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        d3, skip3 = self.down3(d2)
        d4, skip4 = self.down4(d3)
        b = self.bottleneck(d4)

        skip4_g = self.attn4(skip4, b)
        u4 = self.up4(b, skip4_g)

        skip3_g = self.attn3(skip3, u4)
        u3 = self.up3(u4, skip3_g)

        skip2_g = self.attn2(skip2, u3)
        u2 = self.up2(u3, skip2_g)

        u1 = self.up1(u2, skip1)  # skip1 ungated, per paper's design note

        return self.out_conv(u1)

    def get_attention_maps(self):
        """Returns the last cached spatial (alpha) and channel (beta) attention maps
        from each gated level, for visualization (e.g. reproducing paper Fig. 4)."""
        maps = {}
        for name, block in [("level4", self.attn4), ("level3", self.attn3), ("level2", self.attn2)]:
            maps[name] = {"alpha": block.last_alpha, "beta": block.last_beta}
        return maps


def build_model(attention_type: Optional[str], in_channels: int = 1, num_classes: int = 1,
                 base_channels: int = 32) -> AttentionUNet2D:
    """Factory used by train.py / the Colab notebook to build any of the 4 variants
    by name: None, 'spatial', 'cbam', 'hybrid'."""
    return AttentionUNet2D(in_channels, num_classes, base_channels, attention_type)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count, reported in the paper-style results table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
