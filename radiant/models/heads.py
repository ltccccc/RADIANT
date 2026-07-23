import torch
import torch.nn as nn


class BinaryClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(self.pool(x).flatten(1)).squeeze(1)


class CoxHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(self.pool(x).flatten(1)).squeeze(1)
