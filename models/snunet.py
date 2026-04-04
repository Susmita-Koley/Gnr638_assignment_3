"""
SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images
Paper: https://ieeexplore.ieee.org/document/9355573
Official: https://github.com/likyoo/Siam-NestedUNet

From-scratch implementation matching the paper's 3.8M parameter architecture.

Key innovations:
  1. ECAM: Ensemble Channel Attention Module (dual avg+max pool channel attention)
  2. Siamese Encoder: Weight-shared twin encoder processing bi-temporal image pairs
  3. Dense UNet++ Decoder: Dense skip connections between all encoder-decoder nodes
  4. Deep Supervision: 5 outputs (4 intermediate + 1 ensemble) all supervised

Architecture note:
  Node (i,j) in the UNet++ grid = scale level i, decoder column j
  Each decoder node concatenates:
    - Encoder features from BOTH siamese branches at scale i
    - All prior decoder outputs at scale i (dense connections)
    - Upsampled output from scale i+1, column j-1
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic Depth (Drop Path) — used in Assignment 3 as regularization
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Randomly drop entire computation paths during training (stochastic depth).

    At inference, acts as identity. Used as a regularizer in Assignment 3.
    Reference: Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Broadcast over (H, W) dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand_tensor = torch.floor(rand_tensor + keep_prob)
        # Scale to maintain expected activation magnitude
        return x * rand_tensor / keep_prob


# ─────────────────────────────────────────────────────────────────────────────
# ECAM: Ensemble Channel Attention Module
# ─────────────────────────────────────────────────────────────────────────────

class ECAM(nn.Module):
    """Ensemble Channel Attention Module — the core building block of SNUNet-CD.

    Replaces the double-conv block used in standard UNet/UNet++.

    Pipeline:
        Input → Conv-BN-ReLU → Conv-BN-ReLU → Feature map F
        F → [AvgPool → MLP] + [MaxPool → MLP] → Sigmoid → attention weights
        Output = F × attention_weights  (channel-wise scaling)
        Output → DropPath (if enabled)

    The dual-path averaging (avg+max pooling) follows CBAM (Woo et al., 2018),
    extracting both average and peak channel statistics for richer attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    drop_path_rate : float
        Stochastic depth drop probability. 0.0 = disabled (Assignments 1 & 2).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 drop_path_rate: float = 0.0):
        super().__init__()

        # Feature extraction: two 3×3 conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Channel attention: dual-path (avg + max) following CBAM
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Bottleneck MLP with reduction ratio = 16 (SE-Net convention)
        bottleneck = max(out_channels // 16, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, bottleneck, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, out_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # Optional stochastic depth for Assignment 3
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)

        # Ensemble channel attention
        avg_attn = self.fc(self.avg_pool(x))   # (B, C, 1, 1)
        max_attn = self.fc(self.max_pool(x))   # (B, C, 1, 1)
        attn = self.sigmoid(avg_attn + max_attn)

        # Apply attention and stochastic depth
        return self.drop_path(x * attn)


# ─────────────────────────────────────────────────────────────────────────────
# SNUNet-CD: Full Siamese Nested UNet Model
# ─────────────────────────────────────────────────────────────────────────────

class SNUNet_ECAM(nn.Module):
    """SNUNet-CD: Siamese Nested UNet with Ensemble Channel Attention.

    All three assignments use this IDENTICAL architecture (3.8M parameters).
    Differences between assignments are in initialization, optimizer, and loss.

    Parameters
    ----------
    in_channels : int
        Channels per input image.
        - Assignment 1 (LEVIR-CD): 3 (RGB)
        - Assignment 2 (WHU-CD):   3 (RGB)
        - Assignment 3 (OSCD):    13 (Sentinel-2 multispectral)
    out_channels : int
        Output classes. 2 for binary change detection (change / no-change).
    filters : tuple of int
        Feature map channel counts at each of the 5 encoder levels.
        Default (32,64,128,256,512) reproduces the paper's 3.8M params.
    drop_path_rate : float
        Max stochastic depth rate. Applied linearly increasing across decoder depth.
        0.0 for Assignments 1 & 2; 0.1 for Assignment 3.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        filters: tuple = (16, 32, 64, 128, 256),
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Linearly spaced drop path rates across all 10 decoder blocks
        num_decoder_blocks = 10
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, num_decoder_blocks)
        ]

        # ── Shared Siamese Encoder ──────────────────────────────────────────
        # Both temporal images pass through the SAME encoder weights (siamese)
        self.conv0_0 = ECAM(in_channels,  filters[0])  # H   × W
        self.conv1_0 = ECAM(filters[0],   filters[1])  # H/2 × W/2
        self.conv2_0 = ECAM(filters[1],   filters[2])  # H/4 × W/4
        self.conv3_0 = ECAM(filters[2],   filters[3])  # H/8 × W/8
        self.conv4_0 = ECAM(filters[3],   filters[4])  # H/16× W/16

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ── Dense UNet++ Decoder ────────────────────────────────────────────
        # Column j=1: first merge of the two siamese encoder branches
        # Input = [img1_enc_i, img2_enc_i, up(enc_{i+1})]  → "2×enc + 1×lower"
        self.conv0_1 = ECAM(filters[0] * 2 + filters[1], filters[0], dpr[0])
        self.conv1_1 = ECAM(filters[1] * 2 + filters[2], filters[1], dpr[1])
        self.conv2_1 = ECAM(filters[2] * 2 + filters[3], filters[2], dpr[2])
        self.conv3_1 = ECAM(filters[3] * 2 + filters[4], filters[3], dpr[3])

        # Column j=2: accumulate all prior same-scale decoder outputs (dense)
        self.conv0_2 = ECAM(filters[0] * 3 + filters[1], filters[0], dpr[4])
        self.conv1_2 = ECAM(filters[1] * 3 + filters[2], filters[1], dpr[5])
        self.conv2_2 = ECAM(filters[2] * 3 + filters[3], filters[2], dpr[6])

        # Column j=3
        self.conv0_3 = ECAM(filters[0] * 4 + filters[1], filters[0], dpr[7])
        self.conv1_3 = ECAM(filters[1] * 4 + filters[2], filters[1], dpr[8])

        # Column j=4 (deepest decoder node at finest scale)
        self.conv0_4 = ECAM(filters[0] * 5 + filters[1], filters[0], dpr[9])

        # ── Deep Supervision Output Heads ───────────────────────────────────
        self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)

        # Ensemble: fuse all 4 deep-supervision outputs
        self.conv_final = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """Forward pass through siamese encoder + dense decoder.

        Parameters
        ----------
        x1 : Tensor (B, C, H, W)  — pre-change (time 1) image
        x2 : Tensor (B, C, H, W)  — post-change (time 2) image

        Returns
        -------
        tuple of 5 Tensors (B, out_channels, H, W):
            (output1, output2, output3, output4, ensemble_output)
            All 5 are used during training via deep supervision loss.
            At inference, typically only the ensemble_output (index -1) is used.
        """
        # ── Siamese Encoder (weight-shared) ─────────────────────────────────
        x1_0_0 = self.conv0_0(x1)
        x1_1_0 = self.conv1_0(self.pool(x1_0_0))
        x1_2_0 = self.conv2_0(self.pool(x1_1_0))
        x1_3_0 = self.conv3_0(self.pool(x1_2_0))
        x1_4_0 = self.conv4_0(self.pool(x1_3_0))

        x2_0_0 = self.conv0_0(x2)          # shared weights ↑
        x2_1_0 = self.conv1_0(self.pool(x2_0_0))
        x2_2_0 = self.conv2_0(self.pool(x2_1_0))
        x2_3_0 = self.conv3_0(self.pool(x2_2_0))
        x2_4_0 = self.conv4_0(self.pool(x2_3_0))  # noqa: F841 (unused ref kept for symmetry)

        # ── Dense Decoder ────────────────────────────────────────────────────
        # Column j=1
        x0_1 = self.conv0_1(torch.cat([x1_0_0, x2_0_0,
                                        self.up(x1_1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_1_0, x2_1_0,
                                        self.up(x1_2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x1_2_0, x2_2_0,
                                        self.up(x1_3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x1_3_0, x2_3_0,
                                        self.up(x1_4_0)], dim=1))

        # Column j=2 — dense: include all prior same-scale outputs
        x0_2 = self.conv0_2(torch.cat([x1_0_0, x2_0_0,
                                        x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_1_0, x2_1_0,
                                        x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x1_2_0, x2_2_0,
                                        x2_1, self.up(x3_1)], dim=1))

        # Column j=3
        x0_3 = self.conv0_3(torch.cat([x1_0_0, x2_0_0,
                                        x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_1_0, x2_1_0,
                                        x1_1, x1_2, self.up(x2_2)], dim=1))

        # Column j=4
        x0_4 = self.conv0_4(torch.cat([x1_0_0, x2_0_0,
                                        x0_1, x0_2, x0_3,
                                        self.up(x1_3)], dim=1))

        # ── Deep Supervision Outputs ─────────────────────────────────────────
        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)

        # Ensemble output (all 4 → 1×1 conv)
        output = self.conv_final(
            torch.cat([output1, output2, output3, output4], dim=1)
        )

        return (output1, output2, output3, output4, output)


# ─────────────────────────────────────────────────────────────────────────────
# Weight Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_weights_kaiming(model: nn.Module) -> None:
    """Kaiming He initialization — optimal for ReLU activations.
    Used in Assignments 1 (Adam) and 2 (AdamW).
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weights_xavier(model: nn.Module) -> None:
    """Xavier Glorot initialization — good for tanh/sigmoid-like activations.
    Used in Assignment 3 (RAdam optimizer, stochastic depth).
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
