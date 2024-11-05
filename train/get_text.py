from datasets import load_dataset
from pathlib import Path


ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

ds = ds["train"][1100:1200]

text = "".join([x for x in ds["text"]])
with open("data/val_text.txt", "w+", encoding="utf-8") as file:
    file.write(text)
pass
