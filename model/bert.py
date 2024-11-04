import torch
import torch.nn as nn
import torch.nn.functional as F


class BERP(nn.Module):
    def __init__(self, magic_layer: nn.Module, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.magic_layer = magic_layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor):
        # tokens: [seq_len]

        # [seq_len, embedding_dim]
        embeddings = self.embedding(tokens)

        # [seq_len, embedding_dim]
        transformed = self.magic_layer(embeddings)

        # [seq_len, vocab_size]
        projected = self.projection(transformed)

        # [seq_len, vocab_size]
        predictions = F.softmax(projected, dim=-1)

        return predictions


if __name__ == "__main__":
    magic_layer = nn.Linear(768, 768)
    bert = BERP(magic_layer, 30522, 768)
    tokens = torch.randint(0, 30522, (10,))
    print(bert(tokens))
