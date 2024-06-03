import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, dtype=torch.float16),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, dtype=torch.float16),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)