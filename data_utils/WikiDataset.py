import torch
from torch.utils.data import Dataset
from more_itertools import windowed
import sentencepiece as spm
import re


class WikiDataset(Dataset):
    def __init__(
        self,
        corpus_path: str = "data/full_train_text.txt",
        context_window=512,
        spm_model_path="tokenizer/tknz_30000.model",
    ):
        self.tokenizer = spm.SentencePieceProcessor(model_file=spm_model_path)

        with open(corpus_path, "r", encoding="utf-8") as f:
            text = [
                row
                for row in f.readlines()
                if not re.search(r"^\s*=\s*.*?\s*=\s*", row)
            ]
        corpus = "".join(text)

        tokens = self.tokenizer.encode(corpus, out_type=int)
        self.windows_tkns = list(windowed(tokens, context_window))
        self.vocab_size = self.tokenizer.get_piece_size()

    def __len__(self):
        return len(self.windows_tkns)

    def __getitem__(self, index):
        return self.windows_tkns[index]

    def collate_fn(self, list_of_seq: list[list[int]]):
        seq_tensor = torch.tensor(list_of_seq, dtype=torch.long)

        mask = torch.rand(seq_tensor.shape) > 0.15
        masked_seq = torch.where(mask, seq_tensor, torch.tensor(0, dtype=torch.long))
        return masked_seq, seq_tensor


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = WikiDataset()

    dataloader = DataLoader(dataset, 5, shuffle=True, collate_fn=dataset.collate_fn)

    for x in dataloader:
        pass
    pass
