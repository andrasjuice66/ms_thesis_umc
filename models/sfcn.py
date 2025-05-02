"""
Modern-style PyTorch implementation of SFCN
(keeps the original computational graph intact).

  • clean, functional blocks
  • adaptive global pooling (no hard-coded input size)
  • bias-free convolutions + Kaiming weight init
  • type hints for readability
  • returns a tensor instead of a 1-element list
"""

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
#                                building blocks                                #
# ----------------------------------------------------------------------------- #
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
        use_pool: bool,
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


# ----------------------------------------------------------------------------- #
#                                      SFCN                                     #
# ----------------------------------------------------------------------------- #
class SFCN(nn.Module):
    """
    Simple Fully-Convolutional Network for 3-D brain-age prediction.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128, 256, 256, 64),
        num_bins: int = 40,
        dropout_p: Union[float, None] = 0.5,
        log_probs: bool = True,
    ) -> None:
        """
        Args
        ----
        channels  : widths of successive Conv blocks (paper default shown).
        num_bins  : number of age bins / classes.
        dropout_p : `None` or float ∈ (0, 1); keeps behaviour of original paper (0.5).
        log_probs : if True, apply `log_softmax` so loss can be KL-Div / NLL.
        """
        super().__init__()
        self.log_probs = log_probs

        # --------------------------- feature extractor -------------------------- #
        feats: list[nn.Module] = []
        in_ch = 1  # MRI is single-channel

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

        # ------------------------------- classifier ----------------------------- #
        cls_layers: list[nn.Module] = [nn.AdaptiveAvgPool3d(1)]  # global pooling
        if dropout_p:
            cls_layers.append(nn.Dropout(p=dropout_p))
        cls_layers.append(nn.Conv3d(in_ch, num_bins, kernel_size=1))
        self.classifier = nn.Sequential(*cls_layers)

        # weight init (optional but recommended)
        self._init_weights()

    # --------------------------------------------------------------------- #
    #                               utilities                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _init_weights(module):
        for m in module.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # --------------------------------------------------------------------- #
    #                                forward                                #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 1, D, H, W)
        returns:
            (N, num_bins) log-probs (if `log_probs`) or raw logits (else).
        """
        x = self.feature_extractor(x)          # (N, C, d, h, w)
        x = self.classifier(x)                 # (N, num_bins, 1, 1, 1)
        x = torch.flatten(x, 1)                # squeeze spatial dims
        return F.log_softmax(x, dim=1) if self.log_probs else x