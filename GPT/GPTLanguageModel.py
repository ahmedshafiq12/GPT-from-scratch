import torch
import torch.nn as nn
from torch.nn import functional as F
from GPT.Block import Block


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device="auto"):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd, dtype=torch.float16)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd, dtype=torch.float16)
        self.blocks = nn.Sequential(*[Block(self.n_embd, n_head=self.n_head) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, dtype=torch.float16)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -self.block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index

    @torch.no_grad()
    def estimate_loss(self, eval_iters, val_encoded, block_size, batch_size):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                if val_encoded.size(0) > block_size:
                    ix = torch.randint(0, val_encoded.size(0) - block_size, (batch_size,))
                    x = torch.stack([val_encoded[i:i + block_size] for i in ix])
                    y = torch.stack([val_encoded[i + 1:i + block_size + 1] for i in ix])
                else:
                    raise ValueError("Dataset size is too small for the requested block and batch sizes.")

                logits, loss = self.forward(x.to(self.device), y.to(self.device))
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out

    def train_model(self, train_encoded, val_encoded, max_iters, eval_iters, learning_rate, batch_size):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        for iteration in range(max_iters):
            print(iteration)
            if iteration % eval_iters == 0:
                losses = self.estimate_loss(eval_iters, val_encoded, self.block_size, batch_size)
                print(f"step: {iteration}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

            if train_encoded.size(0) > self.block_size:
                ix = torch.randint(0, train_encoded.size(0) - self.block_size, (batch_size,))
                x = torch.stack([train_encoded[i:i + self.block_size] for i in ix])
                y = torch.stack([train_encoded[i + 1:i + self.block_size + 1] for i in ix])
            else:
                raise ValueError("Dataset size is too small for the requested block and batch sizes.")

            logits, loss = self.forward(x.to(self.device), y.to(self.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()