from datasets import load_dataset
from pathlib import Path
import re

DATASET = "validation"


ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
ds = ds[DATASET]


def is_not_title(row):
    text = row["text"]
    pattern = r"^\s*=\s*.*?\s*=\s*"
    return not re.search(pattern, text)


text = " ".join(ds.filter(is_not_title)["text"])
# text = " ".join(ds["text"])

with open(f"data/full_{DATASET}_text.txt", "w+", encoding="utf-8") as file:
    file.write(text)
pass
