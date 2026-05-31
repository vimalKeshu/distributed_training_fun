#!/usr/bin/env python3
"""Download TinyStories data for small pretraining experiments.

The script has two modes:

- subset: streams TinyStories from Hugging Face and writes token-capped train
  and validation text files.
- full: downloads the original TinyStories train/validation text files.

Token counts are measured with the selected Hugging Face tokenizer. The default
uses GPT-2 because it is widely available and good enough for planning data
budgets before training a project-specific tokenizer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "roneneldan/TinyStories"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "tinystories"


@dataclass
class SplitStats:
    split: str
    output_path: str
    stories: int
    tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser(
        "subset",
        help="Stream TinyStories and write token-capped train/valid text files.",
    )
    subset.add_argument(
        "--target-train-tokens",
        type=int,
        default=50_000_000,
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
        help="Hugging Face dataset id to stream.",
    )
    subset.add_argument(
        "--train-split",
        default="train",
        help="Dataset split to use for training subset.",
    )
    subset.add_argument(
        "--valid-split",
        default="validation",
        help="Dataset split to use for validation subset.",
    )

    full = subparsers.add_parser(
        "full",
        help="Download original TinyStories train/valid text files.",
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
    full.add_argument(
        "--include-v2",
        action="store_true",
        help="Also download the GPT-4-only V2 train/valid text files.",
    )

    return parser.parse_args()


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def iter_texts(dataset_id: str, split: str) -> Iterable[str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, streaming=True)
    for row in dataset:
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            yield text.strip()


def write_subset(
    *,
    dataset_id: str,
    split: str,
    tokenizer: Any,
    target_tokens: int,
    output_path: Path,
) -> SplitStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    stories = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for text in iter_texts(dataset_id, split):
            token_count = count_tokens(tokenizer, text)
            if token_count == 0:
                continue
            handle.write(text)
            handle.write("\n\n")
            total_tokens += token_count
            stories += 1
            if total_tokens >= target_tokens:
                break

    return SplitStats(
        split=split,
        output_path=str(output_path),
        stories=stories,
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

    train_name = f"train_{args.target_train_tokens // 1_000_000}M_tokens.txt"
    valid_name = f"valid_{args.target_valid_tokens // 1_000_000}M_tokens.txt"

    train_stats = write_subset(
        dataset_id=args.dataset_id,
        split=args.train_split,
        tokenizer=tokenizer,
        target_tokens=args.target_train_tokens,
        output_path=output_dir / train_name,
    )
    valid_stats = write_subset(
        dataset_id=args.dataset_id,
        split=args.valid_split,
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
    from huggingface_hub import hf_hub_download

    args.output_dir.mkdir(parents=True, exist_ok=True)

    filenames = ["TinyStories-train.txt", "TinyStories-valid.txt"]
    if args.include_v2:
        filenames.extend(["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"])

    downloaded = []
    for filename in filenames:
        path = hf_hub_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            filename=filename,
            local_dir=args.output_dir,
        )
        downloaded.append(path)

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
