"""
SFCN-Reg  –  Fully-convolutional network for 3-D brain-age regression.

Differences from the original SFCN:
    • configurable input channels, dropout, and (noop) attention flag
    • final head returns a single scalar age (float) per sample
    • constructor accepts the exact keys passed from YAML
"""

from __future__ import annotations
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────── building blocks ─────────────────────── #
class ConvBlock(nn.Sequential):
    """
    Conv3d → BatchNorm3d → (optional MaxPool3d) → ReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        padding: int,
        use_pool: bool = True,
    ) -> None:
        layers = [
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_pool:
            # insert MaxPool *before* ReLU
            layers.insert(2, nn.MaxPool3d(kernel_size=2, stride=2))

        super().__init__(*layers)


# ─────────────────────────  model  ──────────────────────────── #
class SFCN(nn.Module):
    """
    Fully-convolutional network for brain-age **regression**.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for most MRI, 3 for multi-modal, …).
    dropout_rate : float | None
        If `None` → no dropout; else applied before final conv.
    use_attention : bool
        Accepted for YAML compatibility; currently a no-op.
    channels : Sequence[int]
        Widths of successive Conv blocks.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        dropout_rate: Union[float, None] = 0.3,
        use_attention: bool = False,            # accepted but unused
        channels: Sequence[int] = (32, 64, 128, 256, 256, 64),
        **_ignored,                             # swallow extra YAML keys (e.g. 'type')
    ) -> None:
        super().__init__()

        # ------------------ feature extractor ------------------ #
        feats: list[nn.Module] = []
        in_ch = in_channels
        for idx, out_ch in enumerate(channels):
            last = idx == len(channels) - 1
            feats.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=1 if last else 3,
                    padding=0 if last else 1,
                    use_pool=not last,
                )
            )
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*feats)

        # ---------------------- head --------------------------- #
        head: list[nn.Module] = [nn.AdaptiveAvgPool3d(1)]  # global pooling
        if dropout_rate:
            head.append(nn.Dropout(dropout_rate))
        head.append(nn.Conv3d(in_ch, 1, kernel_size=1))    # (N,1,1,1,1)
        self.head = nn.Sequential(*head)

        # weight initialisation
        self.apply(self._init_weights)

    # ---------------------------------------------------------- #
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, C, D, H, W)

        Returns
        -------
        (N, 1) predicted age (float).
        """
        x = self.feature_extractor(x)
        x = self.head(x)          # (N,1,1,1,1)
        return x.flatten(1)       # (N,1)