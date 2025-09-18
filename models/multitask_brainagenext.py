import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

# ------------------------------
# Simple 3D Up-sampling block
# ------------------------------
class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm='bn', dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        norm_layer = nn.BatchNorm3d if norm == 'bn' else nn.InstanceNorm3d
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

# ------------------------------
# Simple decoder without skips
# ------------------------------
class SegmentationDecoder3D(nn.Module):
    """
    A lightweight decoder that upsamples the last encoder feature map
    back to the input spatial size using N upsampling stages.

    This does NOT use skip connections (to avoid touching the encoder).
    """
    def __init__(
        self,
        in_channels: int,          # channels of the encoder's final feature map (often your feature_size, e.g., 512)
        num_classes: int = 1,
        num_upsamples: int = 4,    # how many 2x upsamplings to perform
        start_channels: int = None, # if None, start_channels = in_channels
        end_channels: int = 32,    # end of decoder channel width before 1x1 head
        norm: str = 'bn',
        dropout: float = 0.0
    ):
        super().__init__()
        if start_channels is None:
            start_channels = in_channels

        # Plan a smooth channel schedule from in_channels down to end_channels
        chs = [start_channels]
        for i in range(num_upsamples):
            # halve each step but don't go below end_channels
            next_ch = max(end_channels, chs[-1] // 2)
            chs.append(next_ch)

        self.blocks = nn.ModuleList([
            UpBlock3D(chs[i], chs[i+1], norm=norm, dropout=dropout)
            for i in range(num_upsamples)
        ])
        self.head = nn.Conv3d(chs[-1], num_classes, kernel_size=1)

    def forward(self, features, target_spatial_shape=None):
        """
        features: [B, C, D', H', W'] from encoder
        target_spatial_shape: tuple (D, H, W) to ensure final size
        """
        x = features
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x)

        # If shapes don't match exactly (due to odd sizes), adjust by interpolation
        if target_spatial_shape is not None and logits.shape[2:] != target_spatial_shape:
            logits = F.interpolate(logits, size=target_spatial_shape, mode="trilinear", align_corners=False)
        return logits

# ------------------------------
# Multi-task wrapper model
# ------------------------------
class MultiTaskBrainAgeNeXt(nn.Module):
    """
    Wraps your existing BrainAgeNeXt's encoder and regression head,
    and adds a 3D segmentation decoder. Keeps checkpoint compatibility.
    """
    def __init__(
        self,
        brain_age_model: nn.Module,   # an instance of your BrainAgeNeXt with weights loaded
        num_seg_classes: int = 1,
        seg_num_upsamples: int = 4,
        seg_end_channels: int = 32,
        seg_dropout: float = 0.0,
        norm: str = 'bn'
    ):
        super().__init__()
        # Reuse components from the provided brain age model
        self.in_channels = brain_age_model.in_channels
        self.dropout_rate = brain_age_model.dropout_rate
        self.model_id = brain_age_model.model_id
        self.kernel_size = brain_age_model.kernel_size
        self.deep_supervision = brain_age_model.deep_supervision

        self.mednextv1 = brain_age_model.mednextv1               # encoder
        self.global_avg_pool = brain_age_model.global_avg_pool
        self.regression_fc = brain_age_model.regression_fc       # age head (keep weights)

        # Infer the last feature channel count from regression head input (feature_size)
        # Your regression_fc starts with nn.Linear(feature_size, hidden_size)
        # so we can pull feature_size from there:
        first_linear = None
        for m in self.regression_fc.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        assert first_linear is not None, "Could not infer feature_size from regression head."
        feature_size = first_linear.in_features

        # Build segmentation decoder starting from last encoder feature map
        self.seg_decoder = SegmentationDecoder3D(
            in_channels=feature_size,       # final encoder channels
            num_classes=num_seg_classes,
            num_upsamples=seg_num_upsamples,
            start_channels=feature_size,
            end_channels=seg_end_channels,
            norm=norm,
            dropout=seg_dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.mednextv1(x)
        # Some encoders might return tuple/list (e.g., deep supervision); take the last if so
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats

    def forward(self, x: torch.Tensor):
        # Encoder
        feats = self.encode(x)

        # Age head
        pooled = self.global_avg_pool(feats)
        pooled = torch.flatten(pooled, start_dim=1)
        age_pred = self.regression_fc(pooled).squeeze(-1)

        # Seg head: upsample to original size
        seg_logits = self.seg_decoder(feats, target_spatial_shape=x.shape[2:])
        return {"age": age_pred, "seg": seg_logits}