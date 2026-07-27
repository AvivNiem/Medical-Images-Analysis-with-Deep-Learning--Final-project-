"""
layers3d.py
-----------
3D building blocks for the faithful reproduction of Attention U-Net (Oktay et al.,
2018) with our two attention-gate variants. Everything operates on 5D tensors
(B, C, D, H, W).

Modules
  - ConvBlock3D / Down3D / Up3D        : standard 3D U-Net blocks.
  - GridAttentionGate3D                : the ORIGINAL additive grid attention gate,
                                          ported directly from the authors' public
                                          `grid_attention_layer.py` (_concatenation),
                                          updated to modern PyTorch (F.interpolate,
                                          torch.sigmoid). Uses BOTH x and g.
  - CBAM3D (Channel + Spatial)          : the standard CBAM module in 3D — pure
                                          self-attention on the feature map, ignores g.
  - ContextGatedChannelGate3D           : OUR extension — a channel gate conditioned on
                                          BOTH x and g (the same top-down signal the
                                          spatial gate uses), the 3D analogue of the
                                          2D module in the main project.
  - AttentionGate3D                      : one configurable module wrapping the gating
                                          logic for all four variants (none / spatial /
                                          cbam / hybrid), so the network differs between
                                          variants ONLY in this block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_kaiming(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class ConvBlock3D(nn.Module):
    """Two (Conv3x3x3 -> BN -> ReLU) layers — the basic 3D U-Net feature block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    """ConvBlock3D then max-pool by 2. Returns pre-pool skip and pooled output."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class Up3D(nn.Module):
    """Trilinear upsample by 2, 1x1x1 channel reduction, concat with (gated) skip, ConvBlock3D."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, skip_ch, 1)
        self.conv = ConvBlock3D(skip_ch * 2, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class GridAttentionGate3D(nn.Module):
    """Original additive grid attention gate (Oktay et al. 2018), 3D.

    Ported from the authors' `_GridAttentionBlockND._concatenation`:
        theta_x = theta(x)                          # strided conv, downsamples x
        phi_g   = interpolate(phi(g)) to theta_x    # 1x1x1 conv on gating signal
        f       = ReLU(theta_x + phi_g)
        alpha   = sigmoid(psi(f))                   # single-channel attention map
        alpha   = interpolate(alpha) to x's size
        y       = alpha * x
        out     = W(y)                              # 1x1x1 conv + BN

    Returns (out, alpha). Uses BOTH x and the gating signal g.
    """

    def __init__(self, in_ch, gating_ch, inter_ch=None, sub_sample=2):
        super().__init__()
        inter_ch = inter_ch or max(in_ch // 2, 1)
        self.theta = nn.Conv3d(in_ch, inter_ch, kernel_size=sub_sample, stride=sub_sample, bias=False)
        self.phi = nn.Conv3d(gating_ch, inter_ch, kernel_size=1, bias=True)
        self.psi = nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True)
        self.W = nn.Sequential(nn.Conv3d(in_ch, in_ch, 1), nn.BatchNorm3d(in_ch))
        _init_kaiming(self)

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[2:], mode="trilinear", align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        alpha = torch.sigmoid(self.psi(f))
        alpha = F.interpolate(alpha, size=x.shape[2:], mode="trilinear", align_corners=False)
        return self.W(alpha * x), alpha


class CBAM3DChannel(nn.Module):
    """CBAM channel attention (3D) — self-attention over x's pooled statistics only."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, channels))

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool3d(x, 1).flatten(1))
        mx = self.mlp(F.adaptive_max_pool3d(x, 1).flatten(1))
        beta = torch.sigmoid(avg + mx)
        return beta.view(beta.size(0), beta.size(1), 1, 1, 1)


class CBAM3DSpatial(nn.Module):
    """CBAM spatial attention (3D) — conv over channel-pooled maps of x only."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ContextGatedChannelGate3D(nn.Module):
    """OUR extension (3D): a channel gate conditioned on BOTH x and the gating signal g,
    mirroring the spatial gate — unlike CBAM which uses x only. Initialised to pass all
    channels (beta ~ 1) so it cannot suppress useful features before learning."""

    def __init__(self, in_ch, gating_ch, reduction=8):
        super().__init__()
        hidden = max(in_ch // reduction, 4)
        self.fc_x = nn.Linear(in_ch, hidden, bias=False)
        self.fc_g = nn.Linear(gating_ch, hidden, bias=True)
        self.fc_out = nn.Linear(hidden, in_ch, bias=True)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 3.0)  # sigmoid(3) ~ 0.95 -> open at init

    def forward(self, x, g):
        gap_x = F.adaptive_avg_pool3d(x, 1).flatten(1)
        gap_g = F.adaptive_avg_pool3d(g, 1).flatten(1)
        h = F.relu(self.fc_x(gap_x) + self.fc_g(gap_g), inplace=True)
        beta = torch.sigmoid(self.fc_out(h))
        return beta.view(beta.size(0), beta.size(1), 1, 1, 1)


class AttentionGate3D(nn.Module):
    """Configurable skip-connection gate; the ONLY thing that differs between the four
    model variants.

      attention_type:
        None       -> passthrough (vanilla U-Net)
        "spatial"  -> original grid attention gate (x, g)
        "cbam"     -> spatial gate + CBAM channel & spatial self-attention (control)
        "hybrid"   -> spatial gate + context-gated channel gate (x, g)   [ours]
    """

    def __init__(self, in_ch, gating_ch, attention_type=None):
        super().__init__()
        self.attention_type = attention_type
        if attention_type in ("spatial", "cbam", "hybrid"):
            self.spatial = GridAttentionGate3D(in_ch, gating_ch)
        if attention_type == "cbam":
            self.cbam_c = CBAM3DChannel(in_ch)
            self.cbam_s = CBAM3DSpatial()
        if attention_type == "hybrid":
            self.channel = ContextGatedChannelGate3D(in_ch, gating_ch)
        self.last_alpha = None
        self.last_beta = None

    def forward(self, x, g):
        if self.attention_type is None:
            return x
        x_hat, alpha = self.spatial(x, g)   # W(alpha * x)
        self.last_alpha = alpha.detach()
        if self.attention_type == "cbam":
            beta = self.cbam_c(x_hat); x_hat = x_hat * beta
            gamma = self.cbam_s(x_hat); x_hat = x_hat * gamma
            self.last_beta = beta.detach()
        elif self.attention_type == "hybrid":
            beta = self.channel(x, g); x_hat = x_hat * beta
            self.last_beta = beta.detach()
        return x_hat
