import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .decoders import Decoder, DoseDecoder
from ..data.geometry import distance_map


class RadiantStage2(nn.Module):
    def __init__(self, encoder, widths=(16, 24, 48, 64), checkpointing=True):
        super().__init__()
        self.checkpointing = checkpointing
        self.encoder = encoder
        self.gtv_decoder = Decoder(encoder.embed_dim, 1, widths)
        self.ctv_decoder = Decoder(encoder.embed_dim, 2, widths)
        self.dose_decoder = DoseDecoder(encoder.embed_dim, widths)
        self.encoder_frozen = False

    def set_epoch(self, epoch, frozen_epochs):
        self.encoder_frozen = epoch <= frozen_epochs
        self.encoder.requires_grad_(not self.encoder_frozen)
        if self.encoder_frozen:
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def decode(self, decoder, features, detail):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(decoder, features, detail, use_reentrant=False)
        return decoder(features, detail)

    def forward(
        self,
        ct,
        gtv=None,
        ctv=None,
        distance=None,
        spacing_dhw=None,
        teacher_forcing=False,
    ):
        features = self.encoder(ct)
        gtv_logits = self.decode(self.gtv_decoder, features, ct)
        if teacher_forcing:
            if gtv is None or ctv is None or distance is None:
                raise ValueError(
                    "Reference contours and distance map required for training"
                )
            condition_gtv = gtv
        else:
            condition_gtv = (gtv_logits.sigmoid() >= 0.5).to(ct.dtype)
        ctv_logits = self.decode(
            self.ctv_decoder, features, torch.cat([ct, condition_gtv], 1)
        )
        if teacher_forcing:
            condition_ctv = ctv
            signed = distance
        else:
            if spacing_dhw is None:
                raise ValueError("Physical spacing required for complete cascade")
            condition_ctv = (ctv_logits.sigmoid() >= 0.5).to(ct.dtype)
            masks = condition_ctv[:, 0].detach().cpu().numpy()
            values = []
            for mask, spacing in zip(masks, spacing_dhw.detach().cpu().numpy()):
                values.append(distance_map(mask, spacing))
            signed = torch.as_tensor(
                np.stack(values), device=ct.device, dtype=ct.dtype
            )[:, None]
        dose = self.decode(
            self.dose_decoder, features, torch.cat([ct, condition_ctv, signed], 1)
        )
        return {
            "gtv_logits": gtv_logits,
            "ctv_logits": ctv_logits,
            "dose": dose,
            "condition_ctv": condition_ctv,
        }
