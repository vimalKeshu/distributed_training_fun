#!/usr/bin/env python3
"""Train a small byte-level BPE tokenizer on TinyStories.

A ~4k-vocab subword tokenizer makes a small model dramatically more efficient
than raw bytes: a whole story fits in a few hundred tokens instead of a couple
thousand bytes, so the model spends its capacity on story structure rather than
spelling, and a full story fits inside the context window.

The tokenizer is ByteLevel BPE (GPT-2 style): it is lossless and reversible for
any input, and reserves a single `<|endoftext|>` token used to separate
documents during pretraining and as a stop token at sampling time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT_TOKEN = "<|endoftext|>"
DEFAULT_TRAIN = Path(__file__).resolve().parent / "tinystories" / "train_50M_tokens.txt"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "tinystories" / "tokenizer.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN, help="Text file to train on.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save tokenizer.json.")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Target vocabulary size.")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum pair frequency to merge.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_file.exists():
        raise FileNotFoundError(
            f"Missing {args.train_file}. Download data first, for example:\n"
            "python data/download_tinystories.py subset --target-train-tokens 50000000"
        )

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(args.train_file)], trainer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))

    print(f"trained tokenizer: {args.output}")
    print(f"vocab_size: {tokenizer.get_vocab_size()}")
    print(f"{EOT_TOKEN} id: {tokenizer.token_to_id(EOT_TOKEN)}")


if __name__ == "__main__":
    main()
