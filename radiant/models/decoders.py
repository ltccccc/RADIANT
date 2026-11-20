import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(1, outgoing),
            nn.GELU(),
            nn.Conv3d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(1, outgoing),
        )
        self.skip = (
            nn.Identity()
            if incoming == outgoing and stride == 1
            else nn.Conv3d(incoming, outgoing, 1, stride=stride, bias=False)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class Decoder(nn.Module):
    def __init__(
        self, embed_dim, input_channels, widths=(16, 24, 48, 64), positive=False
    ):
        super().__init__()
        self.down = nn.ModuleList(
            [Residual(input_channels, widths[0])]
            + [Residual(a, b, 2) for a, b in zip(widths, widths[1:])]
        )
        self.project = nn.ModuleList([nn.Conv3d(embed_dim, w, 1) for w in widths])
        self.bottom = Residual(2 * widths[-1], widths[-1])
        self.up = nn.ModuleList(
            [Residual(a + 2 * b, b) for a, b in zip(widths[:0:-1], widths[-2::-1])]
        )
        self.head = nn.Conv3d(widths[0], 1, 1)
        self.positive = positive

    def forward(self, features, detail):
        skips = []
        x = detail
        for layer, project, feature in zip(self.down, self.project, features):
            x = layer(x)
            skips.append(
                torch.cat(
                    [
                        x,
                        F.interpolate(
                            project(feature),
                            size=x.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        ),
                    ],
                    1,
                )
            )
        x = self.bottom(skips[-1])
        for layer, skip in zip(self.up, skips[-2::-1]):
            x = layer(
                torch.cat(
                    [
                        F.interpolate(
                            x,
                            size=skip.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        ),
                        skip,
                    ],
                    1,
                )
            )
        y = self.head(x)
        return F.softplus(y) if self.positive else y
