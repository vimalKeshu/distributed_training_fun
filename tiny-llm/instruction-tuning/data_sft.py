#!/usr/bin/env python3
"""Instruction dataset for TinyStories-Instruct (subword / BPE).

Each `<|endoftext|>`-delimited block lists a few instruction fields followed by
`Story:` and the story body. We treat everything up to and including the
`Story:` marker as the *prompt* and the story body as the *response*. The model
is trained with standard next-token prediction, but the loss is masked on the
prompt tokens (set to -100) so it only learns to generate the story.

Prompt and response are tokenized separately so the mask falls on a clean token
boundary, and an `<|endoftext|>` token is appended so the model learns to stop.
Masked positions use the cross-entropy `ignore_index` of -100, which the model's
`forward` already relies on.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

END_OF_TEXT = "<|endoftext|>"
STORY_MARKER = "Story:"
IGNORE_INDEX = -100
# Padding id is arbitrary: pad sits at the end of each (right-padded) sequence,
# causal attention never lets real tokens attend to it, and its target is -100,
# so it never contributes to the loss.
PAD_ID = 0


@dataclass
class SFTExamples:
    """Tokenized instruction examples and their prompt lengths (in tokens)."""

    ids: list[np.ndarray]
    prompt_lens: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


def iter_blocks(text_path: Path):
    """Yield each `<|endoftext|>`-delimited block, streaming the file line by line.

    The corpus can be multiple GB, so we never load it whole; blocks are
    delimited by an `<|endoftext|>` line and emitted one at a time.
    """
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


def build_examples(
    text_path: Path,
    tokenizer: Any,
    *,
    block_size: int,
    require_story_marker: bool = True,
    max_tokens: int | None = None,
) -> SFTExamples:
    ids_list: list[np.ndarray] = []
    prompt_lens: list[int] = []
    skipped = 0
    total_tokens = 0

    for block in iter_blocks(text_path):
        if max_tokens is not None and total_tokens >= max_tokens:
            break
        marker = block.find(STORY_MARKER)
        if marker == -1:
            if require_story_marker:
                skipped += 1
                continue
            prompt_ids: list[int] = []
            response_ids = tokenizer.encode(block)
        else:
            prompt_text = block[: marker + len(STORY_MARKER)]
            response_text = block[marker + len(STORY_MARKER):]
            prompt_ids = tokenizer.encode(prompt_text)
            response_ids = tokenizer.encode(response_text)

        if tokenizer.eot_id is not None:
            response_ids = response_ids + [tokenizer.eot_id]

        # Vocab is < 65536, so uint16 stores ids losslessly at half the memory
        # of int64 (matters for a multi-GB corpus held as a list of arrays).
        ids = np.array(prompt_ids + response_ids, dtype=np.uint16)
        prompt_len = len(prompt_ids)
        if len(ids) < 2 or prompt_len >= len(ids):
            skipped += 1
            continue
        if prompt_len > block_size:
            # Prompt alone does not fit; there is nothing useful to learn.
            skipped += 1
            continue

        ids_list.append(ids)
        prompt_lens.append(prompt_len)
        total_tokens += len(ids)

    if not ids_list:
        raise ValueError(f"No usable instruction examples parsed from {text_path}")

    capped = max_tokens is not None and total_tokens >= max_tokens
    cap_note = f" (capped at ~{max_tokens:,} tokens)" if capped else ""
    print(
        f"parsed {len(ids_list):,} examples / {total_tokens:,} tokens "
        f"from {text_path.name} (skipped {skipped:,}){cap_note}"
    )
    return SFTExamples(ids=ids_list, prompt_lens=np.array(prompt_lens, dtype=np.int64))


def load_or_build_examples(
    *,
    text_path: Path,
    cache_path: Path,
    tokenizer: Any,
    block_size: int,
    require_story_marker: bool,
    overwrite: bool,
    max_tokens: int | None = None,
) -> SFTExamples:
    if cache_path.exists() and not overwrite:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if (
            cached["block_size"] == block_size
            and cached["require_story_marker"] == require_story_marker
            and cached.get("vocab_size") == tokenizer.vocab_size
            and cached.get("max_tokens") == max_tokens
        ):
            return SFTExamples(ids=cached["ids"], prompt_lens=cached["prompt_lens"])

    if not text_path.exists():
        raise FileNotFoundError(
            f"Missing {text_path}. Download the data first, for example:\n"
            "python data/download_tinystories_instruct.py subset --target-train-tokens 20000000"
        )

    examples = build_examples(
        text_path,
        tokenizer,
        block_size=block_size,
        require_story_marker=require_story_marker,
        max_tokens=max_tokens,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "ids": examples.ids,
                "prompt_lens": examples.prompt_lens,
                "block_size": block_size,
                "require_story_marker": require_story_marker,
                "vocab_size": tokenizer.vocab_size,
                "max_tokens": max_tokens,
            },
            handle,
        )
    return examples


def index_stream(num_examples: int, batch_size: int, seed: int):
    """Infinite stream of shuffled micro-batch index arrays (reshuffles each epoch)."""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(num_examples)
        for start in range(0, num_examples - batch_size + 1, batch_size):
            yield order[start : start + batch_size]


def make_batch(
    examples: SFTExamples,
    indices: np.ndarray,
    *,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded (x, y) tensors with the prompt region masked in y."""
    seqs = [examples.ids[i][: block_size + 1] for i in indices]
    plens = [int(examples.prompt_lens[i]) for i in indices]
    max_len = max(len(s) for s in seqs)
    width = max_len - 1

    x = np.full((len(seqs), width), PAD_ID, dtype=np.int64)
    y = np.full((len(seqs), width), IGNORE_INDEX, dtype=np.int64)

    for row, (seq, plen) in enumerate(zip(seqs, plens)):
        length = len(seq)
        x[row, : length - 1] = seq[:-1]
        # Cast to int64 before masking: ids are stored as uint16, but the mask
        # value IGNORE_INDEX (-100) would wrap to a huge positive id in uint16
        # and slip past cross_entropy's ignore_index, crashing the NLL kernel.
        target = seq[1:].astype(np.int64)
        # Target position i predicts seq[i+1]; mask while seq[i+1] is still
        # part of the prompt, i.e. for i < prompt_len - 1.
        mask_until = max(0, plen - 1)
        target[:mask_until] = IGNORE_INDEX
        y[row, : length - 1] = target

    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    return xt, yt
