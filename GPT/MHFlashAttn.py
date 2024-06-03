import torch
import torch.nn as nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


class MHFlashAttn(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd//n_head, head_size, bias=False, dtype=torch.float16)
        self.query = nn.Linear(n_embd//n_head, head_size, bias=False, dtype=torch.float16)
        self.value = nn.Linear(n_embd//n_head, head_size, bias=False, dtype=torch.float16)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_head):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        x = x.view(B, T, n_head, C//n_head) # (B,T,C) -> (B,T,hs,C/hs)
        # print(x.shape)
        
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        # wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # wei = F.softmax(wei, dim=-1) # (B, T, T)
        # wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        # out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True,
                          window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        return out