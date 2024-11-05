import sentencepiece as spm

# spm.SentencePieceTrainer.train(input="data/text.txt", model_prefix="m", vocab_size=5000)
tkn = spm.SentencePieceProcessor(model_file="m.model")

text = "Cat sat on a cat"
pass
