import torch
import torch.nn as nn
import torch.nn.functional as F
from .Block import Block
from .DataLoader import DataLoader
import pickle
from tqdm import tqdm


class GPTLanguageModel(nn.Module):
    def __init__(self, n_embd, n_layer, n_head, dropout, block_size, batch_size, dataset_path, device="auto"):
        print("🚀 Welcome!! I'm your GPT, developed by Ahmed Shafiq. 🚀")
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 I'm using {self.device} as a device")

        super().__init__()

        self.dataloader = DataLoader(block_size, batch_size, dataset_path)
        vocab_size = self.dataloader.vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index

    @torch.no_grad()
    def estimate_loss(self, eval_iters=10):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.dataloader.get_batch(split, self.device)
                logits, loss = self.forward(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.train()
        return out

    def train_model(self, max_iters, eval_iters, learning_rate=5e-4):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        for iter in tqdm(range(max_iters)):
            if iter % eval_iters == 0:
                losses = self.estimate_loss(eval_iters)
                print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

            # Sample a batch of data
            xb, yb = self.dataloader.get_batch('train', self.device)

            # Evaluate the loss
            logits, loss = self.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(loss.item())

        # Save the model
        self.save_model("model.pkl")
        print("Model saved")

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath, device):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        model.device = device
        model.to(device)
        return model