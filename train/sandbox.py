from pathlib import Path
import sys
import torch

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from model.bert import BERP

EMBEDDING_DIM = 50


# Create mini dictionary
text = "cat sat on a mat"
word_to_idx = {word: idx for idx, word in enumerate(set(text.split()))}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
print(word_to_idx)

identity_matrix = torch.eye(len(word_to_idx))

identity_berp = BERP(identity_matrix, len(word_to_idx), EMBEDDING_DIM)

tokens = torch.tensor([word_to_idx[word] for word in text.split()], dtype=torch.int32)


for i in range(500):
    print(identity_berp(tokens))
