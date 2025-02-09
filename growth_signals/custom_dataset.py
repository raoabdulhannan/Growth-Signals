from datasets import load_dataset
from constants import EMBEDDING_DIM
import torch


class CustomDataset():
    def __init__(self, path="Cohere/wikipedia-22-12-en-embeddings"):
        self.data = load_dataset(path, split="train[:1%]", streaming=True)
        # TODO: Get only titles related to physics
        num_samples = int(0.001 * len(self.data))
        self.data = self.data.shuffle(seed=42) \
            .select(range(num_samples))

        self.embedding_dim = EMBEDDING_DIM
        self.embeddings = torch.tensor(
            [self.pad_or_truncate(d['emb']) for d in self.data],
            dtype=torch.float32
        )
        self.texts = self.data['text']

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

    def pad_or_truncate(self, embedding):
        if len(embedding) < self.embedding_dim:
            return embedding + [0.0] * (self.embedding_dim - len(embedding))
        elif len(embedding) > self.embedding_dim:
            return embedding[:self.embedding_dim]
        return embedding
