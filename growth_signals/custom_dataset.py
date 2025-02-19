from datasets import load_dataset
from constants import EMBEDDING_DIM, BATCH_SIZE
import torch
import numpy as np

class CustomDataset(torch.utils.data.IterableDataset):
    def __init__(self, path="Cohere/wikipedia-22-12-en-embeddings"):
        print("Loading data")
        self.data_iter = load_dataset(path, split="train", streaming=True).shuffle(seed=42)
        print("Loaded and shuffled dataset")
        self.embedding_dim = EMBEDDING_DIM
        self.batch_size = BATCH_SIZE

    def process_batch(self, batch):
        batch_size = len(batch)
        embeddings = np.zeros((batch_size, self.embedding_dim), dtype=np.float32)
        texts = []

        for i, d in enumerate(batch):
            emb = np.array(d['emb'], dtype=np.float32)
            length = min(len(emb), self.embedding_dim)
            embeddings[i, :length] = emb[:length]
            texts.append(d['text'])

        return torch.tensor(embeddings), texts

    def __iter__(self):
        batch = []
        for d in self.data_iter:
            batch.append(d)
            if len(batch) == self.batch_size:
                yield self.process_batch(batch)
                batch = []
        if batch:
            yield self.process_batch(batch)
