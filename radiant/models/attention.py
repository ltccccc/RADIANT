import torch
import torch.nn as nn


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
        self.spatial = nn.Conv3d(2, 1, 7, padding=3)

    def forward(self, x):
        b, c, _, _, _ = x.shape
        avg = x.mean(dim=(2, 3, 4))
        mx = x.amax(dim=(2, 3, 4))
        w = torch.sigmoid(self.channel(avg) + self.channel(mx)).view(b, c, 1, 1, 1)
        x = x * w
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(s))


class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=r, dilation=r), nn.GroupNorm(1, out_channels), nn.GELU()) for r in rates])
        self.project = nn.Sequential(nn.Conv3d(out_channels * len(rates), out_channels, 1), nn.GroupNorm(1, out_channels), nn.GELU())

    def forward(self, x):
        return self.project(torch.cat([block(x) for block in self.blocks], dim=1))
