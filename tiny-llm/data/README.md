# Data

Utilities for downloading and preparing pretraining data.

## TinyStories

Install dependencies:

```bash
pip install -r data/requirements.txt
```

Download a small token-capped subset for early experiments:

```bash
python data/download_tinystories.py subset --target-train-tokens 50000000
```

Download a larger 100M-token subset:

```bash
python data/download_tinystories.py subset --target-train-tokens 100000000
```

Download the original TinyStories train and validation text files:

```bash
python data/download_tinystories.py full
```

By default, downloaded/generated data is written under `data/tinystories/`.

## Tokenizer

Train a small ByteLevel BPE tokenizer on the downloaded train text (used by the
BPE training configs instead of raw bytes):

```bash
python data/train_tokenizer.py --vocab-size 4096
```

Writes `data/tinystories/tokenizer.json` with a `<|endoftext|>` token.

## TinyStories-Instruct

Instruction-tuned companion to TinyStories
([`roneneldan/TinyStoriesInstruct`](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)),
used for instruction fine-tuning. Each example is a `<|endoftext|>`-delimited
block listing instruction fields (`Features:`, `Words:`, `Summary:`,
`Random sentence:`) followed by `Story:` and the story body.

Download a token-capped subset for fine-tuning (block delimiters preserved):

```bash
python data/download_tinystories_instruct.py subset --target-train-tokens 20000000
```

Download the original TinyStories-Instruct train and validation text files:

```bash
python data/download_tinystories_instruct.py full
```

By default, generated subset files are written under `data/tinystories_instruct/`.

