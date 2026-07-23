import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(target > 0, prob, 1 - prob)
        alpha_t = torch.where(target > 0, torch.full_like(target, self.alpha), torch.full_like(target, 1 - self.alpha))
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()
