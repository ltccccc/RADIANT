import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 7, padding=3, groups=channels),
            nn.GroupNorm(1, channels),
            nn.Conv3d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv3d(channels * 4, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class DoseBranch3D(nn.Module):
    def __init__(self, output_channels, widths=(64, 128, 256)):
        super().__init__()
        stages = []
        incoming = 1
        for i, width in enumerate(widths):
            stages.append(
                nn.Sequential(
                    nn.Conv3d(incoming, width, 3, stride=2 if i < 2 else 1, padding=1),
                    nn.GroupNorm(1, width),
                    nn.GELU(),
                    ConvNeXtBlock3D(width),
                )
            )
            incoming = width
        self.stages = nn.ModuleList(stages)
        self.project = nn.Conv3d(widths[-1], output_channels, 1)

    def forward(self, dose, shape):
        x = dose
        for stage in self.stages:
            x = (
                checkpoint(stage, x, use_reentrant=False)
                if self.training and torch.is_grad_enabled()
                else stage(x)
            )
        return F.interpolate(
            self.project(x), size=shape, mode="trilinear", align_corners=False
        )


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels)
        )
        self.spatial = nn.Conv3d(2, 1, 7, padding=3)

    def forward(self, x):
        channel = self.channel(x.mean((2, 3, 4))) + self.channel(x.amax((2, 3, 4)))
        x = x * channel.sigmoid()[:, :, None, None, None]
        spatial = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * self.spatial(spatial).sigmoid()


class ASPP3D(nn.Module):
    def __init__(self, incoming, outgoing, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(incoming, outgoing, 3, padding=rate, dilation=rate),
                    nn.GroupNorm(1, outgoing),
                    nn.GELU(),
                )
                for rate in rates
            ]
        )
        self.project = nn.Sequential(
            nn.Conv3d(outgoing * len(rates), outgoing, 1),
            nn.GroupNorm(1, outgoing),
            nn.GELU(),
        )

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], 1))


class RadiantStage3(nn.Module):
    def __init__(
        self,
        encoder,
        task="classification",
        use_immunotherapy=False,
        dose_widths=(64, 128, 256),
        fusion_channels=512,
        hidden_dim=512,
        dropout=0.3,
    ):
        super().__init__()
        if task not in ["classification", "survival"]:
            raise ValueError("task must be classification or survival")
        self.encoder = encoder.requires_grad_(False)
        self.encoder.eval()
        self.task = task
        self.use_immunotherapy = bool(use_immunotherapy)
        self.dose_branch = DoseBranch3D(encoder.embed_dim, dose_widths)
        self.cbam = CBAM3D(2 * encoder.embed_dim, reduction=16)
        self.aspp = ASPP3D(2 * encoder.embed_dim, fusion_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        incoming = fusion_channels + int(self.use_immunotherapy)
        self.head = (
            nn.Sequential(
                nn.Linear(incoming, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if task == "classification"
            else nn.Linear(incoming, 1, bias=False)
        )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, ct, dose, immunotherapy=None):
        if ct.shape != dose.shape or ct.ndim != 5 or ct.shape[1] != 1:
            raise ValueError("CT and generated dose must be aligned Bx1xDxHxW tensors")
        ct_feature = self.encoder(ct)[-1]
        dose_feature = self.dose_branch(dose, ct_feature.shape[2:])
        x = torch.cat([ct_feature, dose_feature], 1)
        if self.training and torch.is_grad_enabled():
            x = checkpoint(self.cbam, x, use_reentrant=False)
            x = checkpoint(self.aspp, x, use_reentrant=False)
        else:
            x = self.aspp(self.cbam(x))
        pooled = self.pool(x).flatten(1)
        if self.use_immunotherapy:
            if immunotherapy is None or immunotherapy.numel() != len(ct):
                raise ValueError("An immunotherapy value is required for each patient")
            z = immunotherapy.to(device=pooled.device, dtype=pooled.dtype).reshape(
                -1, 1
            )
            if not torch.isfinite(z).all() or not ((z == 0) | (z == 1)).all():
                raise ValueError("Immunotherapy must be 0 or 1")
            pooled = torch.cat([pooled, z], 1)
        return self.head(pooled).squeeze(1)
