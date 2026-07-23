import torch
import torch.nn as nn
from .swin3d import SwinTransformer3D
from .decoders import MAEDecoder3D


class RadiantMAE(nn.Module):
    def __init__(self, encoder=None, patch_size=(4, 4, 4), mask_ratio=0.75, embed_dim=384, encoder_kwargs=None):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        self.encoder = encoder or SwinTransformer3D(patch_size=patch_size, embed_dim=embed_dim, **encoder_kwargs)
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = MAEDecoder3D(self.encoder.feature_dims, patch_size)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x):
        tokens, resolution = self.encoder.patch_embed(x)
        b, l, c = tokens.shape
        mask = torch.rand(b, l, device=x.device) < self.mask_ratio
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, l, c), tokens)
        features = []
        for stage in self.encoder.stages:
            tokens, resolution, feat = stage(tokens, resolution)
            features.append(feat)
        recon = self.decoder(features).clamp(0, 1)
        return recon, mask
