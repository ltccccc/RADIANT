import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoderCore3D(nn.Module):
    def __init__(self, feature_dims, base=128):
        super().__init__()
        dims = list(feature_dims)
        self.input_projection = ConvBlock3D(dims[-1], base, kernel_size=1)
        self.skip_projections = nn.ModuleList([ConvBlock3D(d, base, kernel_size=1) for d in reversed(dims[:-1])])
        self.upsampling_blocks = nn.ModuleList([ConvBlock3D(base * 2, base) for _ in dims[:-1]])

    def forward(self, features):
        x = self.input_projection(features[-1])
        for feat, skip_projection, block in zip(reversed(features[:-1]), self.skip_projections, self.upsampling_blocks):
            x = F.interpolate(x, size=feat.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip_projection(feat)], dim=1)
            x = block(x)
        return x


class SegmentationDecoder3D(nn.Module):
    def __init__(self, feature_dims, out_channels=1, condition_channels=0, base=128):
        super().__init__()
        self.projection = UNetDecoderCore3D(feature_dims, base)
        self.condition_projection = ConvBlock3D(condition_channels, base, kernel_size=1) if condition_channels else None
        self.refinement = nn.Sequential(
            ConvBlock3D(base, base),
            ConvBlock3D(base, base // 2),
        )
        self.head = nn.Conv3d(base // 2, out_channels, 1)

    def forward(self, features, output_size, condition=None):
        x = self.projection(features)
        if self.condition_projection is not None and condition is not None:
            c = F.interpolate(condition, size=x.shape[2:], mode="trilinear", align_corners=False)
            x = x + self.condition_projection(c)
        x = self.refinement(x)
        x = self.head(x)
        return F.interpolate(x, size=output_size, mode="trilinear", align_corners=False)


class DoseDecoder3D(nn.Module):
    def __init__(self, feature_dims, condition_channels=1, base=128):
        super().__init__()
        self.projection = UNetDecoderCore3D(feature_dims, base)
        self.condition_projection = ConvBlock3D(condition_channels, base, kernel_size=1)
        self.regression_decoder = ConvBlock3D(base, base)
        self.refinement = nn.Sequential(
            ConvBlock3D(base, base),
            ConvBlock3D(base, base // 2),
        )
        self.head = nn.Conv3d(base // 2, 1, 1)

    def forward(self, features, output_size, condition):
        x = self.projection(features)
        c = F.interpolate(condition, size=x.shape[2:], mode="trilinear", align_corners=False)
        x = x + self.condition_projection(c)
        x = self.regression_decoder(x)
        x = self.refinement(x)
        x = self.head(x)
        x = F.interpolate(x, size=output_size, mode="trilinear", align_corners=False)
        return F.softplus(x)


class MAEDecoder3D(nn.Module):
    def __init__(self, feature_dims, patch_size=(4, 4, 4)):
        super().__init__()
        dims = list(feature_dims)
        self.up_blocks = nn.ModuleList()
        for src, dst in zip(reversed(dims[1:]), reversed(dims[:-1])):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(src, dst, 2, 2),
                    ConvBlock3D(dst * 2, dst),
                    ConvBlock3D(dst, dst),
                )
            )
        self.head = nn.Sequential(
            ConvBlock3D(dims[0], dims[0]),
            nn.ConvTranspose3d(dims[0], 1, patch_size, patch_size),
        )

    def forward(self, features):
        x = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            x = block[0](x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block[1](x)
            x = block[2](x)
        return self.head(x)
