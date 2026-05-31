#!/usr/bin/env python3
"""Download TinyStories-Instruct data for instruction fine-tuning.

The dataset is the instruction-tuned companion to TinyStories from the same
authors (Eldan & Li). Each example is a `<|endoftext|>`-delimited block that
lists a few instruction fields (some subset of `Features:`, `Words:`,
`Summary:`, `Random sentence:`) followed by `Story:` and the story body.

The script has two modes:

- subset: writes token-capped train and validation text files, keeping the
  `<|endoftext|>` block delimiters intact so the SFT loader can split on them.
- full: downloads the original TinyStories-Instruct train/validation text files.

Token counts are measured with the selected Hugging Face tokenizer (GPT-2 by
default) only for budgeting; the model itself is byte-level.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "roneneldan/TinyStoriesInstruct"
TRAIN_FILE = "TinyStories-Instruct-train.txt"
VALID_FILE = "TinyStories-Instruct-valid.txt"
END_OF_TEXT = "<|endoftext|>"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "tinystories_instruct"


@dataclass
class SplitStats:
    split: str
    output_path: str
    blocks: int
    tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser(
        "subset",
        help="Write token-capped train/valid instruction text files.",
    )
    subset.add_argument(
        "--target-train-tokens",
        type=int,
        default=20_000_000,
        help="Approximate number of tokenizer tokens to write for train.",
    )
    subset.add_argument(
        "--target-valid-tokens",
        type=int,
        default=1_000_000,
        help="Approximate number of tokenizer tokens to write for validation.",
    )
    subset.add_argument(
        "--tokenizer-name",
        default="gpt2",
        help="Tokenizer used only for counting tokens.",
    )
    subset.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated subset files.",
    )
    subset.add_argument(
        "--dataset-id",
        default=DATASET_ID,
        help="Hugging Face dataset id to download from.",
    )

    full = subparsers.add_parser(
        "full",
        help="Download original TinyStories-Instruct train/valid text files.",
    )
    full.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "full",
        help="Directory for full downloaded text files.",
    )
    full.add_argument(
        "--dataset-id",
        default=DATASET_ID,
        help="Hugging Face dataset id to download from.",
    )

    return parser.parse_args()


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def download_source(dataset_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=filename)
    )


def iter_blocks(source_path: Path) -> Iterable[str]:
    """Yield `<|endoftext|>`-delimited blocks while streaming line by line."""
    buffer: list[str] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == END_OF_TEXT:
                block = "".join(buffer).strip("\n")
                if block.strip():
                    yield block
                buffer = []
            else:
                buffer.append(line)
    tail = "".join(buffer).strip("\n")
    if tail.strip():
        yield tail


def write_subset(
    *,
    source_path: Path,
    split: str,
    tokenizer: Any,
    target_tokens: int,
    output_path: Path,
) -> SplitStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    blocks = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for block in iter_blocks(source_path):
            token_count = count_tokens(tokenizer, block)
            if token_count == 0:
                continue
            handle.write(block)
            handle.write(f"\n{END_OF_TEXT}\n")
            total_tokens += token_count
            blocks += 1
            if total_tokens >= target_tokens:
                break

    return SplitStats(
        split=split,
        output_path=str(output_path),
        blocks=blocks,
        tokens=total_tokens,
    )


def write_manifest(output_dir: Path, manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_subset(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_source = download_source(args.dataset_id, TRAIN_FILE)
    valid_source = download_source(args.dataset_id, VALID_FILE)

    train_name = f"train_{args.target_train_tokens // 1_000_000}M_tokens.txt"
    valid_name = f"valid_{args.target_valid_tokens // 1_000_000}M_tokens.txt"

    train_stats = write_subset(
        source_path=train_source,
        split="train",
        tokenizer=tokenizer,
        target_tokens=args.target_train_tokens,
        output_path=output_dir / train_name,
    )
    valid_stats = write_subset(
        source_path=valid_source,
        split="validation",
        tokenizer=tokenizer,
        target_tokens=args.target_valid_tokens,
        output_path=output_dir / valid_name,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "subset",
        "dataset_id": args.dataset_id,
        "tokenizer_name": args.tokenizer_name,
        "target_train_tokens": args.target_train_tokens,
        "target_valid_tokens": args.target_valid_tokens,
        "splits": [asdict(train_stats), asdict(valid_stats)],
    }
    write_manifest(output_dir, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))


def run_full(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for filename in [TRAIN_FILE, VALID_FILE]:
        source = download_source(args.dataset_id, filename)
        target = args.output_dir / filename
        target.write_bytes(source.read_bytes())
        downloaded.append(str(target))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full",
        "dataset_id": args.dataset_id,
        "output_dir": str(args.output_dir),
        "files": downloaded,
    }
    write_manifest(args.output_dir, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "subset":
        run_subset(args)
    elif args.command == "full":
        run_full(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
