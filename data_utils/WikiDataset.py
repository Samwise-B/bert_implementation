import torch
from torch.utils.data import Dataset
from more_itertools import windowed
import sentencepiece as spm
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        
        self.tnsr_tkns = torch.tensor(tokens, dtype=torch.int16, requires_grad=False)

        mask = torch.rand(self.tnsr_tkns.shape) > 0.15
        
        self.masked_tkns = torch.where(mask, self.tnsr_tkns, 0)

        self.windows = self.tnsr_tkns.unfold(dimension=0, size=context_window, step=1)
        self.masked_windows = self.masked_tkns.unfold(dimension=0, size=context_window, step=1)

        # self.windows_tkns = list(windowed(tokens, context_window))
        self.vocab_size = self.tokenizer.get_piece_size()

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, index):
        return self.windows[index], self.masked_windows[index]

    def collate_fn(self, list_of_seq: list[torch.Tensor]):
        input, masked = zip(*list_of_seq)

        seq_tensor = torch.stack(input, dim=0).to(device=device, dtype=torch.long)

        # mask = torch.rand(seq_tensor.shape, device=device) > 0.15
        # masked_seq = torch.where(mask, seq_tensor, 0)

        masked_seq = torch.stack(masked, dim=0).to(device=device, dtype=torch.long)
        return masked_seq, seq_tensor


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = WikiDataset()

    dataloader = DataLoader(dataset, 5, shuffle=True, collate_fn=dataset.collate_fn)

    for x in dataloader:
        pass
    pass
