"""
Modern, *drop-in equivalent* implementation of the original SFCN
(PAC 2019 brain-age “soft-classification” model).

Differences from the 2019 reference:
    • rewritten with type hints and cleaner sub-modules
    • `AdaptiveAvgPool3d(1)` (shape-agnostic) instead of a hard-coded
      `AvgPool3d([5, 6, 5])`  – behaviour is identical if the input shape
      is the same (it still outputs a single voxel per feature map)
    • configurable age range: one bin per integer year, here 20 – 85 ⇒ 66 bins
    • returns a *tensor* of log-probabilities rather than a list

Everything else – convolution order, ReLU placement, `log_softmax`
in the forward pass – is exactly as in the original paper/codebase.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────── helper block ───────────────────────── #
class _ConvBlock(nn.Sequential):
    """
    Conv3d → BatchNorm3d → (optional MaxPool3d) → ReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        padding: int = 1,
        use_pool: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
        ]
        if use_pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


# ──────────────────────────   model   ────────────────────────── #
class SFCNOriginal(nn.Module):
    """
    SFCN for brain-age prediction as a soft-classification problem.

    Each output channel is a *log-probability* over an integer-age bin.
    Training is normally done with `nn.KLDivLoss(reduction="batchmean")`
    against Gaussian soft labels centred on the true age.

    Parameters
    ----------
    in_channels     : number of MRI modalities (default 1)
    channels        : widths of the 6 convolutional blocks
    age_min, age_max: inclusive age range; one bin per integer year
    dropout_rate    : if 0 → no dropout before the classifier
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 256, 64),
        age_min: int = 20,
        age_max: int = 85,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        if age_max < age_min:
            raise ValueError("age_max must be ≥ age_min")
        self.age_min = age_min
        self.age_max = age_max
        self.n_bins: int = age_max - age_min + 1  # ← 66 for 20-85

        # ------------- feature extractor (6 conv blocks) ------------- #
        extractor: list[nn.Module] = []
        in_ch = in_channels
        n_blocks = len(channels)
        for i, out_ch in enumerate(channels):
            last = i == n_blocks - 1
            extractor.append(
                _ConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=1 if last else 3,
                    padding=0 if last else 1,
                    use_pool=not last,
                )
            )
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*extractor)

        # ------------------------ classifier ------------------------- #
        cls: list[nn.Module] = [
            nn.AdaptiveAvgPool3d(1),                      # → (N, C, 1, 1, 1)
        ]
        if dropout_rate > 0:
            cls.append(nn.Dropout(dropout_rate))
        cls.append(nn.Conv3d(in_ch, self.n_bins, kernel_size=1))  # logits
        self.classifier = nn.Sequential(*cls)

        # weight init
        self.apply(self._init_weights)

    # --------------------------------------------------------------- #
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # --------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor of shape (N, C_in, D, H, W)

        Returns
        -------
        log_p : tensor (N, n_bins)  – log-probabilities over age bins
        """
        x = self.feature_extractor(x)
        x = self.classifier(x)                     # (N, n_bins, 1, 1, 1)
        log_p = F.log_softmax(x.flatten(1), dim=1) # flatten & normalise
        return log_p

    # --------------------------------------------------------------- #
    def expected_age(self, log_p: torch.Tensor) -> torch.Tensor:
        """
        Convert log-probabilities to the scalar expected age.

        Parameters
        ----------
        log_p : (N, n_bins)  – output of `forward`

        Returns
        -------
        agê  : (N,) float tensor
        """
        p = log_p.exp()
        bins = torch.arange(self.age_min, self.age_max + 1, device=log_p.device)
        return (p * bins).sum(dim=1)