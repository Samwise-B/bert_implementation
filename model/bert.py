import torch
import torch.nn as nn
import torch.nn.functional as F


class BERP(nn.Module):
    def __init__(self, magic_layer: nn.Module, vocab_size: int, embedding_dim: int):
        super().__init__()

        # Must return tensor of the same shape
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


class Naive(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, embeddings: torch.Tensor):
        # Embeddings: [batch_size, seq_len, embedding_dim]

        # [batch_size, seq_len, seq_len]
        x = torch.bmm(embeddings, embeddings.transpose(-1, -2))

        # [batch_size, seq_len, embedding_dim]
        transformed = torch.bmm(x, embeddings)

        return self.relu(transformed)


class OwnSingleHeadTransformer(nn.Module):
    def __init__(self, embedding_dim: int, ff_dim: int):
        super().__init__()
        self.embed_dim = embedding_dim
        self.scaling_fac = self.embed_dim ** (1 / 2)
        self.M_q = nn.Linear(embedding_dim, embedding_dim)
        self.M_k = nn.Linear(embedding_dim, embedding_dim)
        self.M_v = nn.Linear(embedding_dim, embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, embeddings: torch.Tensor):
        # Embeddings: [batch_size, seq_len, embedding_dim]
        Q = self.M_q(embeddings)

        # [batch_size, seq_len, embedding_dim]
        K = self.M_k(embeddings)

        # [batch_size, seq_len, seq_len]
        A_prime = torch.bmm(Q, K.transpose(-1, -2)) / self.scaling_fac

        # [batch_size, seq_len, seq_len]
        A = F.softmax(A_prime, dim=-1)

        # [batch_size, seq_len, emb_dim]
        V = self.M_v(embeddings)

        # [batch_size, seq_len, embedding_dim]
        attn_emb = torch.bmm(A, V)

        # [batch_size, seq_len, embedding_dim]
        return self.ff(attn_emb)


if __name__ == "__main__":
    magic_layer = nn.Linear(768, 768)
    bert = BERP(magic_layer, 30522, 768)
    tokens = torch.randint(0, 30522, (10,))
    print(bert(tokens))
