import torch
import torch.nn as nn
from GPT.MHFlashAttn import MHFlashAttn


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, dropout, n_embd):
        super().__init__(dropout)
        # self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.mha_flash_attn = MHFlashAttn(head_size)
        # self.proj = nn.Linear(head_size * num_heads, n_embd, dtype=torch.float16)
        self.proj = nn.Linear(head_size, n_embd, dtype=torch.float16)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.mha_flash_attn(x)
        out = self.dropout(self.proj(out))
        return out
