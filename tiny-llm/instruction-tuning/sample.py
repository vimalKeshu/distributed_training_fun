#!/usr/bin/env python3
"""Sample a story from an instruction fine-tuned checkpoint.

Build a TinyStories-Instruct style prompt from the provided instruction fields,
then let the model generate the story that follows `Story:`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer_util import EOT_TOKEN, load_tokenizer

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "data" / "tinystories" / "tokenizer.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from an instruction-tuned checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--summary", default=None, help="Summary instruction field.")
    parser.add_argument("--words", default=None, help="Comma-separated words to include.")
    parser.add_argument("--features", default=None, help="Comma-separated features, e.g. 'Dialogue'.")
    parser.add_argument("--random-sentence", default=None, help="A sentence to appear in the story.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Raw prompt overriding the instruction fields (must end at 'Story:').",
    )
    parser.add_argument("--max-new-tokens", type=int, default=800)
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


def build_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    lines = []
    if args.features:
        lines.append(f"Features: {args.features}")
    if args.words:
        lines.append(f"Words: {args.words}")
    if args.summary:
        lines.append(f"Summary: {args.summary}")
    if args.random_sentence:
        lines.append(f"Random sentence: {args.random_sentence}")
    if not lines:
        lines.append("Summary: A short, simple story for young children.")
    lines.append("Story:")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt_text = build_prompt(args)
    prompt = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long).unsqueeze(0).to(device)
    output = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(output[0].cpu().tolist())
    # Stop at the first end-of-text delimiter if the model produced one.
    print(text.split(EOT_TOKEN)[0])


if __name__ == "__main__":
    main()
