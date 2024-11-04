from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from model.bert import BERP

EMBEDDING_DIM = 50


# Create mini dictionary
text = "cat sat on a mat mat mat"
word_to_idx = {word: idx for idx, word in enumerate(set(text.split()))}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
print(word_to_idx)

identity_matrix = lambda x: x

identity_berp = BERP(identity_matrix, len(word_to_idx), EMBEDDING_DIM)

tokens = torch.tensor([word_to_idx[word] for word in text.split()], dtype=torch.long)


optimiser = optim.SGD(identity_berp.parameters())
optimiser.zero_grad()

criterion = nn.CrossEntropyLoss()

for i in range(100_000):
    preds = identity_berp(tokens)

    loss: torch.Tensor = criterion(preds, tokens)

    loss.backward()

    optimiser.step()
    optimiser.zero_grad()

    print(loss.item(), end="\r")

# Validate
preds = identity_berp(tokens)

pred_tokens = ...

# if loss.item() < 0.5:
