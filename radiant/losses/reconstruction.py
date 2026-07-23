import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim3d(x, y, kernel=7):
    pad = kernel // 2
    ux = F.avg_pool3d(x, kernel, 1, pad)
    uy = F.avg_pool3d(y, kernel, 1, pad)
    vx = F.avg_pool3d(x * x, kernel, 1, pad) - ux * ux
    vy = F.avg_pool3d(y * y, kernel, 1, pad) - uy * uy
    vxy = F.avg_pool3d(x * y, kernel, 1, pad) - ux * uy
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-6)


class MAEReconstructionLoss(nn.Module):
    def __init__(self, alpha_ssim=0.5):
        super().__init__()
        self.alpha_ssim = alpha_ssim

    def forward(self, recon, target, mask=None):
        if mask is None:
            l1 = torch.abs(recon - target).mean()
            ssim_loss = 1.0 - ssim3d(recon, target).mean()
            return l1 + self.alpha_ssim * ssim_loss
        mask = mask.to(recon.dtype)
        denom = mask.sum().clamp_min(1.0)
        l1 = (torch.abs(recon - target) * mask).sum() / denom
        ssim_loss = ((1.0 - ssim3d(recon, target)) * mask).sum() / denom
        return l1 + self.alpha_ssim * ssim_loss
