import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Convolutional block for the U-Net.
    Corresponds to the sequence of convolutions and activations at each level of the SynthSeg Keras model.
    """
    def __init__(self, in_channels, out_channels, n_convs=2, conv_size=3, activation='elu'):
        super().__init__()
        
        # In the Keras SynthSeg model, bias is used in convolutions, and no normalization layer is applied.
        # This implementation follows that, using bias=True and no normalization layers.
        # The activation function is also configurable to match the original model.
        
        layers = []
        for i in range(n_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(nn.Conv3d(in_ch, out_channels, kernel_size=conv_size, padding='same', bias=True))
            if activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            elif activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f"Activation function {activation} is not supported.")
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    """
    Encoder part of the U-Net architecture.
    """
    def __init__(self, chs=(24, 48, 96, 192, 384), n_convs=2):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = 1  # assuming single-channel MRI
        for i, ch in enumerate(chs):
            self.downs.append(ConvBlock(prev_ch, ch, n_convs=n_convs))
            if i < len(chs) - 1:
                self.pools.append(nn.MaxPool3d(2))
            prev_ch = ch

    def forward(self, x):
        feats = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                feats.append(x)
                x = self.pools[i](x)
        return x, feats

class Decoder(nn.Module):
    """
    Decoder part of the U-Net architecture.
    """
    def __init__(self, n_classes, chs=(384, 192, 96, 48, 24), n_convs=2):
        super().__init__()
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.ups.append(nn.ConvTranspose3d(chs[i], chs[i+1], kernel_size=2, stride=2))
            self.convs.append(ConvBlock(chs[i+1] * 2, chs[i+1], n_convs=n_convs))
        
        # The final layer produces logits, corresponding to the Keras model's final linear convolution.
        self.out = nn.Conv3d(chs[-1], n_classes, kernel_size=1)
        # Softmax is applied to get probabilities, as in the original model.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, encoder_feats):
        # The decoder uses reversed skip connections from the encoder
        for i in range(len(self.convs)):
            x = self.ups[i](x)
            enc_f = encoder_feats[-(i + 1)]
            x = torch.cat([x, enc_f], dim=1)
            x = self.convs[i](x)
        
        x = self.out(x)
        x = self.softmax(x)
        return x

class SynthSeg(nn.Module):
    """
    PyTorch implementation of the SynthSeg U-Net model.
    The architecture is based on the Keras implementation and configured with parameters from the SynthSeg prediction script.
    """
    def __init__(self, n_classes, n_levels=5, n_convs=2, init_feat=24, feat_mult=2):
        super().__init__()
        
        # Determine channel numbers based on the initial feature count and multiplier
        encoder_chs = [init_feat]
        for _ in range(n_levels - 1):
            encoder_chs.append(int(encoder_chs[-1] * feat_mult))
        
        encoder_chs = tuple(encoder_chs)
        decoder_chs = tuple(reversed(encoder_chs))

        self.encoder = Encoder(chs=encoder_chs, n_convs=n_convs)
        self.decoder = Decoder(n_classes, chs=decoder_chs, n_convs=n_convs)

    def forward(self, x):
        deepest_features, skip_connections = self.encoder(x)
        segmentation = self.decoder(deepest_features, skip_connections)
        return segmentation 