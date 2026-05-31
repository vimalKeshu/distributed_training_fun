#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer_util import EOT_TOKEN, load_tokenizer

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "data" / "tinystories" / "tokenizer.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from an Assignment 1 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


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
    device = select_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long).unsqueeze(0).to(device)
    output = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(output[0].cpu().tolist())
    print(text.split(EOT_TOKEN)[0])


if __name__ == "__main__":
    main()

