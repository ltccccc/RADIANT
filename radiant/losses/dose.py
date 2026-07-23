import torch.nn as nn
import torch.nn.functional as F


class DoseRegressionLoss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)
