import torch
import torch.nn as nn
from GPT.MultiHeadAttention import MultiHeadAttention
from GPT.FeedForward import FeedForward


class Block(nn.Module):
    def __init__(self, n_embd, n_head, batch_size, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size = n_embd // n_head
        head_size = n_embd
        x_size = torch.tensor([batch_size, block_size, n_head, n_embd])
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # self.ln1 = nn.LayerNorm(n_embd, dtype=torch.float16)
        # self.ln2 = nn.LayerNorm(n_embd, dtype=torch.float16)
        # self.ln1 = nn.LayerNorm(x_size[1:])
        # self.ln2 = nn.LayerNorm(x_size[1:])
        self.ln1 = nn.LayerNorm(n_embd, head_size, dtype=torch.float16)
        self.ln2 = nn.LayerNorm(n_embd, head_size, dtype=torch.float16)
        
    def forward(self, x):
        y = self.sa(x)
        print(y.shape)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x