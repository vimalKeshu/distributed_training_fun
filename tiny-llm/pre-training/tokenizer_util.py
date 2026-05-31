"""Thin wrapper around the trained ByteLevel BPE tokenizer.

Centralizes encode/decode and the `<|endoftext|>` id so training, sampling, and
evaluation all agree on the vocabulary.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

EOT_TOKEN = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing tokenizer {path}. Train it first:\n"
                "python data/train_tokenizer.py --vocab-size 4096"
            )
        self.tokenizer = Tokenizer.from_file(str(path))
        self.eot_id = self.tokenizer.token_to_id(EOT_TOKEN)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def load_tokenizer(path: str | Path) -> BPETokenizer:
    return BPETokenizer(path)
