import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, conv_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, conv_size, padding='same', bias=False),
            nn.InstanceNorm3d(out_ch),  # Changed from BatchNorm3d
            nn.ELU(inplace=True),
            nn.Conv3d(out_ch, out_ch, conv_size, padding='same', bias=False),
            nn.InstanceNorm3d(out_ch),  # Changed from BatchNorm3d
            nn.ELU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Encoder(nn.Module):
    def __init__(self, chs=(24, 48, 96, 192, 384)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = 1        # assuming single-channel MRI
        for i, ch in enumerate(chs):
            self.downs.append(ConvBlock(prev, ch))
            if i < len(chs) - 1: # no pooling on last level
                self.pools.append(nn.MaxPool3d(2))
            prev = ch
            
    def forward(self, x):
        feats = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                feats.append(x)
                x = self.pools[i](x)
        return x, feats          # deepest feature + skip feats

class SegDecoder(nn.Module):
    def __init__(self, n_classes, chs=(384, 192, 96, 48, 24)):
        super().__init__()
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(chs)-1):
            self.ups.append(nn.ConvTranspose3d(chs[i], chs[i+1], 2, stride=2))
            # Input to conv is concatenated features from upsampling and skip connection
            self.convs.append(ConvBlock(chs[i+1] * 2, chs[i+1]))
        self.out = nn.Conv3d(chs[-1], n_classes, 1)

    def forward(self, x, encoder_feats):
        for up, conv, enc_f in zip(self.ups, self.convs, reversed(encoder_feats)):
            x = up(x)
            x = torch.cat([x, enc_f], dim=1)
            x = conv(x)
        return self.out(x), x   # logits and final decoder features
        

class AgeHead(nn.Module):
    def __init__(self, in_ch, hidden=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),             # [B, C,1,1,1]
            nn.Flatten(),
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.head(x).squeeze(1)  # [B]

class MultiTaskBrainAge(nn.Module):
    def __init__(self, n_classes, encoder_chs=(24, 48, 96, 192, 384)):
        super().__init__()
        self.encoder = Encoder(chs=encoder_chs)
        # in_ch for heads = last encoder channels (e.g. 384)
        decoder_chs = tuple(reversed(encoder_chs))
        self.seg_head = SegDecoder(n_classes, chs=decoder_chs)
        
        # New input channels for AgeHead will be from encoder bottleneck and decoder final layer
        age_head_in_ch = encoder_chs[-1] + decoder_chs[-1]
        self.age_head = AgeHead(in_ch=age_head_in_ch)

    def forward(self, x):
        deepest, skips = self.encoder(x)
        seg_logits, decoder_final_feat = self.seg_head(deepest, skips)

        # Downsample decoder features to match bottleneck's spatial dimensions
        downsampler = nn.AdaptiveAvgPool3d(deepest.shape[2:])
        downsampled_decoder_feat = downsampler(decoder_final_feat)
        
        # Concatenate features along the channel dimension
        combined_features = torch.cat([deepest, downsampled_decoder_feat], dim=1)

        age_pred   = self.age_head(combined_features)
        return seg_logits, age_pred
    
