import torch
import numpy as np
from datasets import load_dataset
from constants import EMBEDDING_DIM, BATCH_SIZE

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path="Cohere/wikipedia-22-12-en-embeddings", num_rows=100000, shuffle=False):
        print("Loading data")
        self.dataset = load_dataset(path, split="train", streaming=False)
        
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=42)
            print(f"Loaded and shuffled dataset")
        else:
            print(f"Loaded dataset without shuffling")
        
        self.dataset = self.dataset.select(range(num_rows))
        print(f"Selected {num_rows} rows from dataset")
        
        self.embedding_dim = EMBEDDING_DIM
        self.batch_size = BATCH_SIZE
        self.dataset.set_format(type="numpy", columns=["emb", "text", "title"])

    def process_batch(self, batch):
        embeddings_list = [
            np.pad(
                d['emb'][:self.embedding_dim].astype(np.float32),
                (0, max(0, self.embedding_dim - len(d['emb']))),
                mode='constant'
            )
            for d in batch
        ]
        texts = [d['text'] for d in batch]
        titles = [d['title'] for d in batch]
        embeddings = np.stack(embeddings_list, axis=0)
        return torch.from_numpy(embeddings), texts, titles

    def __iter__(self):
        batch = []
        for d in self.dataset:
            batch.append(d)
            if len(batch) == self.batch_size:
                yield self.process_batch(batch)
                batch = []
        if batch:
            yield self.process_batch(batch)

    def __getitem__(self, index):
        sample = self.dataset[index]
        emb = sample['emb']
        if len(emb) < self.embedding_dim:
            emb = np.pad(emb, (0, self.embedding_dim - len(emb)), mode='constant')
        else:
            emb = emb[:self.embedding_dim]
        embedding = torch.from_numpy(emb.astype(np.float32))
        text = sample['text']
        title = sample['title']
        return embedding, text, title

    def __len__(self):
        return len(self.dataset)