#!/usr/bin/env python3
"""Prompt-only dataset for on-policy distillation.

Unlike SFT (which trains on the ground-truth story), on-policy distillation only
needs the *prompt* (the instruction fields up to and including `Story:`). The
student generates its own continuation from each prompt, and the teacher grades
those continuations. So here we parse each `<|endoftext|>`-delimited block,
extract just the prompt, tokenize it, and store the prompt token ids.

The file is streamed line by line (it can be multiple GB) and tokens are stored
as uint16 (vocab < 65536) to keep memory small.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

END_OF_TEXT = "<|endoftext|>"
STORY_MARKER = "Story:"


@dataclass
class Prompts:
    """Tokenized instruction prompts (each ends at the `Story:` marker)."""

    ids: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.ids)


def iter_blocks(text_path: Path):
    """Yield each `<|endoftext|>`-delimited block, streaming the file line by line."""
    buffer: list[str] = []
    with text_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == END_OF_TEXT:
                block = "".join(buffer).strip()
                buffer = []
                if block:
                    yield block
            else:
                buffer.append(line)
    tail = "".join(buffer).strip()
    if tail:
        yield tail


def build_prompts(
    text_path: Path,
    tokenizer: Any,
    *,
    block_size: int,
    min_response_tokens: int = 64,
    max_tokens: int | None = None,
) -> Prompts:
    """Parse prompts, keeping those that leave room for a generated response.

    A prompt is the text up to and including `Story:`. We drop prompts that are
    so long there is no room to generate at least `min_response_tokens` within
    the model's `block_size` (prompt + response must fit the context window).
    """
    ids_list: list[np.ndarray] = []
    skipped = 0
    total_tokens = 0
    max_prompt_len = block_size - min_response_tokens

    for block in iter_blocks(text_path):
        if max_tokens is not None and total_tokens >= max_tokens:
            break
        marker = block.find(STORY_MARKER)
        if marker == -1:
            skipped += 1
            continue
        prompt_text = block[: marker + len(STORY_MARKER)]
        prompt_ids = tokenizer.encode(prompt_text)
        if len(prompt_ids) < 2 or len(prompt_ids) > max_prompt_len:
            skipped += 1
            continue
        ids_list.append(np.array(prompt_ids, dtype=np.uint16))
        total_tokens += len(prompt_ids)

    if not ids_list:
        raise ValueError(f"No usable prompts parsed from {text_path}")

    capped = max_tokens is not None and total_tokens >= max_tokens
    cap_note = f" (capped at ~{max_tokens:,} tokens)" if capped else ""
    print(
        f"parsed {len(ids_list):,} prompts / {total_tokens:,} tokens "
        f"from {text_path.name} (skipped {skipped:,}){cap_note}"
    )
    return Prompts(ids=ids_list)


def load_or_build_prompts(
    *,
    text_path: Path,
    cache_path: Path,
    tokenizer: Any,
    block_size: int,
    min_response_tokens: int,
    overwrite: bool,
    max_tokens: int | None = None,
) -> Prompts:
    if cache_path.exists() and not overwrite:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if (
            cached["block_size"] == block_size
            and cached.get("min_response_tokens") == min_response_tokens
            and cached.get("vocab_size") == tokenizer.vocab_size
            and cached.get("max_tokens") == max_tokens
        ):
            return Prompts(ids=cached["ids"])

    if not text_path.exists():
        raise FileNotFoundError(
            f"Missing {text_path}. Download the data first, for example:\n"
            "python data/download_tinystories_instruct.py subset --target-train-tokens 20000000"
        )

    prompts = build_prompts(
        text_path,
        tokenizer,
        block_size=block_size,
        min_response_tokens=min_response_tokens,
        max_tokens=max_tokens,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "ids": prompts.ids,
                "block_size": block_size,
                "min_response_tokens": min_response_tokens,
                "vocab_size": tokenizer.vocab_size,
                "max_tokens": max_tokens,
            },
            handle,
        )
    return prompts


def index_stream(num_prompts: int, batch_size: int, seed: int):
    """Infinite stream of shuffled index arrays (reshuffles each epoch)."""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(num_prompts)
        for start in range(0, num_prompts - batch_size + 1, batch_size):
            yield order[start : start + batch_size]
