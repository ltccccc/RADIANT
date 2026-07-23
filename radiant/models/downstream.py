import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin3d import SwinTransformer3D
from .attention import CBAM3D, ASPP3D
from .heads import BinaryClassificationHead, CoxHead


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 7, padding=3, groups=channels),
            nn.GroupNorm(1, channels),
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvNeXtDoseBranch3D(nn.Module):
    def __init__(self, out_channels, channels=(64, 128, 256)):
        super().__init__()
        blocks = []
        in_channels = 1
        for i, channels_i in enumerate(channels):
            stride = 2 if i < 2 else 1
            blocks.extend([
                nn.Conv3d(in_channels, channels_i, 3, stride=stride, padding=1),
                nn.GroupNorm(1, channels_i),
                nn.GELU(),
                ConvNeXtBlock3D(channels_i),
            ])
            in_channels = channels_i
        blocks.append(nn.Conv3d(channels[-1], out_channels, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x, size):
        x = self.net(x)
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
class RadiantDownstream(nn.Module):
    def __init__(self, encoder=None, task="classification", hidden_dim=512, dropout=0.3, aspp_rates=(1, 6, 12, 18), freeze_encoder=True, encoder_kwargs=None):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        self.encoder = encoder or SwinTransformer3D(**encoder_kwargs)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        enc_channels = self.encoder.feature_dims[-1]
        self.dose_branch = ConvNeXtDoseBranch3D(enc_channels)
        fused = enc_channels * 2
        self.cbam = CBAM3D(fused)
        self.aspp = ASPP3D(fused, hidden_dim, aspp_rates)
        self.head = BinaryClassificationHead(hidden_dim, hidden_dim, dropout) if task == "classification" else CoxHead(hidden_dim, hidden_dim, dropout)

    def forward(self, ct, dose):
        ct_feat = self.encoder(ct)[-1]
        dose_feat = self.dose_branch(dose, ct_feat.shape[2:])
        x = torch.cat([ct_feat, dose_feat], dim=1)
        x = self.aspp(self.cbam(x))
        return self.head(x)
