"""
layers.py
---------
Building blocks for the U-Net variants used in this project:

  1. ConvBlock / DownBlock / UpBlock  - standard U-Net encoder/decoder blocks.
  2. SpatialAttentionGate2D           - the ORIGINAL additive attention gate from
                                         Oktay et al. 2018 (Attention U-Net), Eq. 1-2.
                                         Cross-checked against the official reference
                                         implementation (grid_attention_layer.py,
                                         ozan-oktay/Attention-Gated-Networks).
  3. CBAMChannelGate / CBAMSpatialGate - the standard (self-attention only) CBAM
                                         blocks used by the "naive bolt-on" baseline
                                         (reproducing the ASCU-Net / CBAM-AG-UNet
                                         style of combining AG with CBAM).
  4. ContextGatedChannelGate           - OUR proposed extension: a channel-attention
                                         module conditioned on both the local features
                                         x and the coarse-scale gating signal g,
                                         mirroring the spatial gate's own design
                                         instead of using self-attention pooling only.
  5. AttentionBlock2D                  - a single configurable module that wraps the
                                         skip-connection gating logic for all four
                                         model variants (none / spatial / cbam / hybrid),
                                         so models.py can build all variants from one
                                         shared implementation.

All modules operate on 2D tensors shaped (B, C, H, W), since the project uses 2D
axial CT slices rather than full 3D volumes (see docs/reference_notes.md for why).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two (Conv3x3 -> BatchNorm -> ReLU) layers, the basic U-Net feature block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Max-pool by 2, then a ConvBlock. Returns the pre-pool skip features too."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    """Bilinear upsample by 2, concatenate with (possibly gated) skip features, ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.conv = ConvBlock(skip_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SpatialAttentionGate2D(nn.Module):
    """
    Original additive attention gate (Oktay et al. 2018, Eq. 1-2).

    Computes a single-channel spatial attention map alpha in [0, 1] from the local
    skip-connection features x and a coarser-scale gating signal g:

        theta_x = Theta(x)        # 1x1(or strided) conv, downsamples x to g's resolution
        phi_g   = Phi(g)          # 1x1 conv
        f       = ReLU(theta_x + phi_g)
        alpha   = Sigmoid(Psi(f)) # upsampled back to x's resolution

    Returns alpha only (NOT x * alpha) so that AttentionBlock2D can combine it with
    the channel gate before applying the shared output transform, exactly mirroring
    how the original paper applies a single W transform after gating.
    """

    def __init__(self, x_channels: int, g_channels: int, inter_channels: int = None, sub_sample_factor: int = 2):
        super().__init__()
        inter_channels = inter_channels or max(x_channels // 2, 1)
        self.theta = nn.Conv2d(x_channels, inter_channels, kernel_size=sub_sample_factor,
                                stride=sub_sample_factor, bias=False)
        self.phi = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

        # Initialise the gate to start OPEN (pass-through), as the original paper
        # does: "Gating parameters are initialised so that attention gates pass
        # through feature vectors at all spatial locations." We push psi's output
        # strongly positive at init so sigmoid(psi) ~ 1 everywhere, i.e. alpha ~ 1.
        # Without this, the gate can start half-closed and strangle the skip
        # connection early in training, causing divergence to an all-background
        # prediction (observed empirically for the gated variants).
        nn.init.zeros_(self.psi.weight)
        nn.init.constant_(self.psi.bias, 3.0)  # sigmoid(3) ~ 0.95

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta(x)
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[-2:], mode="bilinear", align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        psi_f = torch.sigmoid(self.psi(f))
        alpha = F.interpolate(psi_f, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return alpha  # (B, 1, H, W)


class ContextGatedChannelGate(nn.Module):
    """
    OUR proposed extension.

    A channel-attention module analogous to SpatialAttentionGate2D, but producing a
    per-channel weight beta instead of a per-pixel weight. Crucially, beta is
    conditioned on BOTH the local features x and the coarse-scale gating signal g
    (via global-average-pooled summaries), not on x alone. This is the key
    difference from standard CBAM channel attention, which is pure self-attention
    over x's own statistics and never sees g.

        h    = ReLU( Wxc . GAP(x) + Wgc . GAP(g) )
        beta = Sigmoid( Wout . h )
    """

    def __init__(self, x_channels: int, g_channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(x_channels // reduction, 4)
        self.fc_x = nn.Linear(x_channels, hidden, bias=False)
        self.fc_g = nn.Linear(g_channels, hidden, bias=True)
        self.fc_out = nn.Linear(hidden, x_channels, bias=True)

        # Same pass-through initialisation rationale as the spatial gate: start
        # with beta ~ 1 for every channel so the channel gate does not suppress
        # useful features before it has learned anything.
        nn.init.zeros_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 3.0)  # sigmoid(3) ~ 0.95

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gap_x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, Cx)
        gap_g = F.adaptive_avg_pool2d(g, 1).flatten(1)   # (B, Cg)
        h = F.relu(self.fc_x(gap_x) + self.fc_g(gap_g), inplace=True)
        beta = torch.sigmoid(self.fc_out(h))             # (B, Cx)
        return beta.unsqueeze(-1).unsqueeze(-1)           # (B, Cx, 1, 1)


class CBAMChannelGate(nn.Module):
    """Standard CBAM channel attention: self-attention over x only (no g). Used as
    the 'naive bolt-on' control baseline, reproducing what ASCU-Net / CBAM-AG-UNet
    style papers do."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1).flatten(1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1).flatten(1))
        beta = torch.sigmoid(avg + mx)
        return beta.unsqueeze(-1).unsqueeze(-1)


class CBAMSpatialGate(nn.Module):
    """Standard CBAM spatial attention: 7x7 conv over channel-pooled maps of x only."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class AttentionBlock2D(nn.Module):
    """
    Single configurable skip-connection gate used to build all four model variants
    from one shared implementation.

    attention_type:
      - None       : passthrough, x_hat = x                       (vanilla U-Net)
      - "spatial"  : x_hat = W(x * alpha)                          (original Attention U-Net)
      - "cbam"     : x_hat = W(x * alpha) then CBAM channel+spatial (naive bolt-on baseline)
      - "hybrid"   : x_hat = W(x * alpha * beta_context_gated)      (our proposed method)
    """

    def __init__(self, x_channels: int, g_channels: int, attention_type: str = None):
        super().__init__()
        self.attention_type = attention_type

        if attention_type in ("spatial", "cbam", "hybrid"):
            self.spatial_gate = SpatialAttentionGate2D(x_channels, g_channels)

        if attention_type == "cbam":
            self.cbam_channel = CBAMChannelGate(x_channels)
            self.cbam_spatial = CBAMSpatialGate()

        if attention_type == "hybrid":
            self.channel_gate = ContextGatedChannelGate(x_channels, g_channels)

        if attention_type in ("spatial", "cbam", "hybrid"):
            self.output_transform = nn.Sequential(
                nn.Conv2d(x_channels, x_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(x_channels),
            )

        self.last_alpha = None  # cached for visualization
        self.last_beta = None

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.attention_type is None:
            return x

        alpha = self.spatial_gate(x, g)
        self.last_alpha = alpha.detach()
        x_hat = x * alpha

        if self.attention_type == "cbam":
            beta = self.cbam_channel(x_hat)
            x_hat = x_hat * beta
            gamma = self.cbam_spatial(x_hat)
            x_hat = x_hat * gamma
            self.last_beta = beta.detach()

        elif self.attention_type == "hybrid":
            beta = self.channel_gate(x, g)
            self.last_beta = beta.detach()
            x_hat = x_hat * beta

        return self.output_transform(x_hat)
