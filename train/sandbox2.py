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
from model.bert import Naive, BERP, OwnSingleHeadTransformer, OwnMultiHeadTransformer

EMBEDDING_DIM = 100
FF_DIM = 500

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
    "own-single-head-transformer": OwnSingleHeadTransformer(EMBEDDING_DIM, FF_DIM),
    "own-multi-head-transformer": OwnMultiHeadTransformer(EMBEDDING_DIM, 2, FF_DIM),
    "torch-baseline": nn.TransformerEncoderLayer(
        EMBEDDING_DIM, nhead=1, dim_feedforward=FF_DIM
    ),
    "torch-8-head-baseline": nn.TransformerEncoderLayer(
        EMBEDDING_DIM, nhead=10, dim_feedforward=FF_DIM
    ),
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


for name, layer in magic_layers.items():
    print(f"{name} : {count_parameters(layer)}")
