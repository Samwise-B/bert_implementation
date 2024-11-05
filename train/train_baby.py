from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
import wandb
from tqdm import tqdm

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from data_utils.baby_dataset import BabyDataset
from model.bert import Naive, BERP

dataset = BabyDataset()
val_dataset = BabyDataset(corpus_path="data/val_text.txt")

EMBEDDING_DIM = 10
VOCAB_SIZE = dataset.vocab_size

magic_layers = {
    # "identity": lambda x: x,
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

dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=32, shuffle=True, collate_fn=val_dataset.collate_fn
)

wandb_project = "baby-bert"

criterion = nn.CrossEntropyLoss()

for name, magic_layer in magic_layers.items():

    bert = BERP(magic_layer, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM).to(
        device
    )

    optimizer = torch.optim.Adam(bert.parameters(), lr=1e-3)

    wandb.init(project=wandb_project, name=f"baby-bert-{name}")

    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(20):
        train_loss = []
        for batch in tqdm(dataloader, desc=f"Training {name} {epoch}"):
            optimizer.zero_grad()
            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = bert(inputs)

            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        wandb.log({"loss": sum(train_loss) / len(train_loss)})

        # manual eval
        if not epoch % 1:
            val_loss = []
            for batch in tqdm(val_dataloader, desc=f"Validating {name} {epoch}"):
                with torch.no_grad():
                    _, targets = batch

                    targets = targets.to(device)

                    outputs = bert(targets)

                    loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))

                    val_loss.append(loss.item())

            wandb.log({"val_loss": sum(val_loss) / len(val_loss)})

    wandb.finish()

    demo_text = "Hi! my name is slimshady"
    tokens = dataset.tokenizer.encode(demo_text, out_type=int)
    tokens = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        tokens = tokens.to(device)
        outputs = bert(tokens)

        predictions = outputs.argmax(dim=-1)

        text = dataset.tokenizer.decode(predictions[0].tolist())
        pass
