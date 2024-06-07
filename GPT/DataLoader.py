import torch
import random
import mmap
import os


class DataLoader:
    def __init__(self, block_size, batch_size, dataset_path):
        self.vocab_filepath = os.path.join(dataset_path, "vocab.txt")
        with open(self.vocab_filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.string_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_string = {i: ch for i, ch in enumerate(self.chars)}
        self.block_size = block_size
        self.batch_size = batch_size
        self.train_filepath = os.path.join(dataset_path, "output_train.txt")
        self.val_filepath = os.path.join(dataset_path, "output_val.txt")

    def encode(self, s):
        return [self.string_to_int[c] for c in s]

    def decode(self, l):
        return ''.join([self.int_to_string[i] for i in l])

    def get_random_chunk(self, split):
        filename = self.train_filepath if split == 'train' else self.val_filepath
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Determine the file size and a random position to start reading
                file_size = len(mm)
                start_pos = random.randint(0, file_size - self.block_size * self.batch_size)

                # Seek to the random position and read the block of text
                mm.seek(start_pos)
                block = mm.read(self.block_size * self.batch_size)

                # Decode the block to a string, ignoring any invalid byte sequences
                decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

                # Encode the block to integers
                data = torch.tensor(self.encode(decoded_block), dtype=torch.long)
        return data

    def get_batch(self, split, device):
        data = self.get_random_chunk(split)
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
