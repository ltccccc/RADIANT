import torch
import torch.nn as nn
from .swin3d import SwinTransformer3D
from .decoders import DoseDecoder3D, SegmentationDecoder3D


class RadiantStage2(nn.Module):
    def __init__(self, encoder=None, encoder_kwargs=None):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        self.encoder = encoder or SwinTransformer3D(**encoder_kwargs)
        dims = self.encoder.feature_dims
        self.gtv_decoder = SegmentationDecoder3D(dims)
        self.ctv_decoder = SegmentationDecoder3D(dims, condition_channels=1)
        self.dose_decoder = DoseDecoder3D(dims, condition_channels=1)

    def forward(self, ct):
        features = self.encoder(ct)
        size = ct.shape[2:]
        gtv_logits = self.gtv_decoder(features, size)
        pred_gtv = torch.sigmoid(gtv_logits)
        ctv_logits = self.ctv_decoder(features, size, pred_gtv)
        pred_ctv = torch.sigmoid(ctv_logits)
        dose = self.dose_decoder(features, size, pred_ctv)
        return {"gtv_logits": gtv_logits, "ctv_logits": ctv_logits, "dose": dose, "features": features}
