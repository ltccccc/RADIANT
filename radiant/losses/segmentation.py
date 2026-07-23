import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        target = target.float()
        inter = (prob * target).sum(dim=(1, 2, 3, 4))
        union = prob.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
        dice = 1 - ((2 * inter + 1) / (union + 1)).mean()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.where(target > 0, prob, 1 - prob)
        focal = (self.alpha * (1 - pt).pow(self.gamma) * bce).mean()
        return dice + focal

