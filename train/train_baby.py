from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from data_utils.baby_dataset import BabyDataset
from model.bert import Naive, BERP

dataset = BabyDataset()
val_dataset = BabyDataset(corpus_path="data/val_text.txt")

EMBEDDING_DIM = 10
VOCAB_SIZE = dataset.vocab_size

magic_layers = {
    "identity": lambda x: x,
    "linear": nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
    "non-linear": nn.Sequential(
        nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        nn.ReLU(),
    ),
    "naive": Naive(EMBEDDING_DIM),
    "multi-naive": nn.Sequential(
        Naive(EMBEDDING_DIM),
        Naive(EMBEDDING_DIM),
    ),
}

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

wandb_project = "baby-bert"

for name, magic_layer in magic_layers.items():

    bert = BERP(magic_layer, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)

    optimizer = torch.optim.Adam(bert.parameters(), lr=1e-3)

    wandb.init(project=wandb_project, name=f"baby-bert-{name}")

    # Train
    for epoch in range(100):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch

            outputs = bert(inputs)

            loss = F.cross_entropy(outputs, targets)

            loss.backward()

            optimizer.step()

            wandb.log({"loss": loss})

        # manual eval
        if not epoch % 1:
            for batch in val_dataloader:
                with torch.no_grad():
                    inputs, targets = batch

                    outputs = bert(inputs)

                    loss = F.cross_entropy(outputs, targets)

                    wandb.log({"val_loss": loss})
