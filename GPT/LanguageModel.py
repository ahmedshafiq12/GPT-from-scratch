import torch
import torch.nn as nn
import torch.nn.functional as F
from .Block import Block
from .DataLoader import DataLoader
import pickle
from tqdm import tqdm
import os


class GPTLanguageModel(nn.Module):
    def __init__(self, n_embd, n_layer, n_head, dropout, block_size, batch_size, dict_path="dicts/vocab.txt", dataset_path="", device="auto"):
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.block_size = block_size
        self.batch_size = batch_size
        print("🚀 Welcome!! I'm your ChatGPT, developed by Ahmed Shafiq. 🚀")
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 I'm using {self.device} as a device")

        super().__init__()

        self.dataloader = DataLoader(block_size, batch_size, dict_path, dataset_path, self.device)
        vocab_size = self.dataloader.vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        self.to(self.device)
        self.saving_dir = "weights"
        os.makedirs(self.saving_dir, exist_ok=True)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index.to(self.device)) # (B,T,C)
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

        return logits.to(self.device), loss.to(self.device)

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -self.block_size:].to(self.device)
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1).to(self.device)  # (B, C)
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
                X, Y = self.dataloader.get_batch(split)
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
                print(f"\n ✅ step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
                self.save_model(f"{iter}_epochs_model.pt")

            # Sample a batch of data
            xb, yb = self.dataloader.get_batch('train')

            # Evaluate the loss
            logits, loss = self.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(loss.item())

        # Save the model
        self.save_model("final_model.pt")

    def talk(self, prompt):
        context = torch.tensor(self.dataloader.encode(prompt), dtype=torch.long, device=self.device)
        generated_chars = self.dataloader.decode(self.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
        return generated_chars

    def save_model(self, file_name):
        file_path = os.path.join(self.saving_dir, file_name)
        torch.save(self, file_path)
        print(f"\n 💾 Model: {file_name} saved")

    def load_model(self, file_path):
        self = torch.load(file_path)
        self.to(self.device)