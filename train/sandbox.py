from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from model.bert import BERP

EMBEDDING_DIM = 3


# Create mini dictionary
text = "cat sat on a mat mat mat"
corpus = "cat sat on a mat mat mat date dave visiona sd soi sodoc wodlkc eimdcdi wejfk djdkdi weflewcl"
idx_to_word = {idx: word for idx, word in enumerate(set(corpus.split()), start=1)}
idx_to_word[0] = "<PAD>"
word_to_idx = {word: idx for idx, word in idx_to_word.items()}
print(word_to_idx)

tokens = torch.tensor([word_to_idx[word] for word in text.split()], dtype=torch.long)
identity_matrix = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

identity_berp = BERP(identity_matrix, len(word_to_idx), EMBEDDING_DIM)


optimiser = optim.SGD(identity_berp.parameters())
optimiser.zero_grad()

criterion = nn.CrossEntropyLoss()


import random

for i in range(100_000):

    masked_tokens = tokens.clone()
    masked_tokens[random.randint(0, 6)] = 0

    preds = identity_berp(masked_tokens)

    loss: torch.Tensor = criterion(preds, tokens)

    loss.backward()

    optimiser.step()
    optimiser.zero_grad()

    print(f"{loss.item():.10f} epoch {i}", end="\r")


# Validate
preds = identity_berp(tokens)

pred_tokens = torch.argmax(preds, dim=-1)

sns.heatmap(identity_berp.magic_layer.weight.data.numpy())
plt.show()
pass


# if loss.item() < 0.5:
