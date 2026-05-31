#!/usr/bin/env python3
"""Evaluate masked validation loss for an instruction fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from data_sft import index_stream, load_or_build_examples, make_batch
from model import GPT, GPTConfig
from tokenizer_util import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an instruction-tuned GPT.")
    parser.add_argument("config", type=Path, help="Path to the JSON SFT config.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)

    data_config = config["data"]
    training_config = config["training"]

    device = select_device(training_config["device"])
    dtype_name = training_config.get("dtype", "float32")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    ctx = nullcontext() if device == "cpu" or dtype == torch.float32 else torch.autocast(device_type=device, dtype=dtype)

    out_dir = resolve_path(training_config["out_dir"])
    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model_config = GPTConfig(**checkpoint["model_config"])

    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = load_tokenizer(resolve_path(data_config["tokenizer_path"]))
    examples = load_or_build_examples(
        text_path=resolve_path(data_config["valid_text_path"]),
        cache_path=resolve_path(data_config["cache_dir"]) / "valid_sft.pkl",
        tokenizer=tokenizer,
        block_size=model_config.block_size,
        require_story_marker=data_config.get("require_story_marker", True),
        overwrite=data_config.get("overwrite_cache", False),
    )

    batch_size = int(training_config["batch_size"])
    sampler = index_stream(len(examples), batch_size, config.get("seed", 1337))
    eval_iters = max(training_config["eval_iters"], 512)

    print(f"run_name: {config['run_name']}")
    print(f"device: {device}")
    print(f"parameters: {model.parameter_count() / 1_000_000:.2f}M")
    print(f"valid examples: {len(examples):,}")

    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for index in range(eval_iters):
            batch_indices = next(sampler)
            x, y = make_batch(examples, batch_indices, block_size=model_config.block_size, device=device)
            with ctx:
                _, loss = model(x, y)
            losses[index] = loss.item()

    print(f"valid {losses.mean().item():.4f}")


if __name__ == "__main__":
    main()
