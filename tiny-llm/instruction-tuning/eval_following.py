#!/usr/bin/env python3
"""Instruction-following metrics for an instruction-tuned checkpoint.

Cross-entropy loss (see eval.py) measures next-byte prediction; it does not
tell you whether the model actually *follows* the instruction. This script
generates a story for each validation prompt and measures interpretable
behaviour:

- words_used:      fraction of the requested `Words:` that appear in the story
- sentence_exact:  fraction of examples whose `Random sentence:` appears verbatim
- sentence_fuzzy:  average longest-common-substring coverage of that sentence

Metrics are computed only on the *generated* continuation (the prompt, which
already contains the words and the sentence, is excluded). The same metrics are
reported for the reference stories as a ceiling: that is the score a perfect
model would get, since even the ground-truth stories do not always contain every
requested word verbatim.
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import torch

from data_sft import END_OF_TEXT, STORY_MARKER, iter_blocks
from model import GPT, GPTConfig
from tokenizer_util import load_tokenizer

DEFAULT_TOKENIZER = Path(__file__).resolve().parent.parent / "data" / "tinystories" / "tokenizer.json"

FIELD_PREFIXES = {
    "Features:": "features",
    "Words:": "words",
    "Summary:": "summary",
    "Random sentence:": "random_sentence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--valid-path",
        type=Path,
        required=True,
        help="TinyStories-Instruct validation text file.",
    )
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--show", type=int, default=2, help="Print this many sample generations.")
    parser.add_argument("--seed", type=int, default=1337)
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


def parse_block(block: str) -> dict | None:
    marker = block.find(STORY_MARKER)
    if marker == -1:
        return None
    header = block[:marker]
    fields: dict = {"features": None, "words": [], "summary": None, "random_sentence": None}
    for line in header.splitlines():
        line = line.strip()
        for prefix, key in FIELD_PREFIXES.items():
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if key == "words":
                    fields["words"] = [w.strip().lower() for w in value.split(",") if w.strip()]
                else:
                    fields[key] = value
                break
    fields["prompt"] = block[: marker + len(STORY_MARKER)]
    fields["reference"] = block[marker + len(STORY_MARKER):].strip()
    return fields


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def word_coverage(words: list[str], story: str) -> float | None:
    if not words:
        return None
    lowered = story.lower()
    hits = sum(1 for w in words if w in lowered)
    return hits / len(words)


def sentence_recall(sentence: str | None, story: str) -> tuple[float, float] | tuple[None, None]:
    if not sentence:
        return None, None
    target = normalize(sentence)
    hay = normalize(story)
    if not target:
        return None, None
    exact = 1.0 if target in hay else 0.0
    matcher = difflib.SequenceMatcher(None, target, hay, autojunk=False)
    longest = matcher.find_longest_match(0, len(target), 0, len(hay))
    fuzzy = longest.size / len(target)
    return exact, fuzzy


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.tokenizer)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    examples = [parsed for block in iter_blocks(args.valid_path) if (parsed := parse_block(block))]
    if not examples:
        raise ValueError(f"No usable instruction examples parsed from {args.valid_path}")

    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(len(examples))[: args.num_examples]

    gen_words, gen_exact, gen_fuzzy = [], [], []
    ref_words, ref_exact, ref_fuzzy = [], [], []

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device} | parameters: {model.parameter_count() / 1_000_000:.2f}M")
    print(f"evaluating {len(chosen)} of {len(examples):,} validation examples\n")

    for rank, idx in enumerate(chosen):
        example = examples[idx]
        prompt_ids = torch.tensor(tokenizer.encode(example["prompt"]), dtype=torch.long)
        prompt_len = prompt_ids.numel()
        prompt = prompt_ids.unsqueeze(0).to(device)
        output = model.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        generated = tokenizer.decode(output[0, prompt_len:].cpu().tolist()).split(END_OF_TEXT)[0].strip()

        words = example["words"]
        sentence = example["random_sentence"]

        cov = word_coverage(words, generated)
        if cov is not None:
            gen_words.append(cov)
            ref_words.append(word_coverage(words, example["reference"]))

        g_exact, g_fuzzy = sentence_recall(sentence, generated)
        if g_exact is not None:
            gen_exact.append(g_exact)
            gen_fuzzy.append(g_fuzzy)
            r_exact, r_fuzzy = sentence_recall(sentence, example["reference"])
            ref_exact.append(r_exact)
            ref_fuzzy.append(r_fuzzy)

        if rank < args.show:
            print("=" * 72)
            print(example["prompt"].strip())
            print("--- generated ---")
            print(generated[:600])
            print(f"[words_used={cov} sentence_exact={g_exact} sentence_fuzzy={None if g_fuzzy is None else round(g_fuzzy, 2)}]")
            print()

    print("=" * 72)
    print(f"{'metric':<18}{'generated':>12}{'reference':>12}")
    print(f"{'words_used':<18}{mean(gen_words):>12.3f}{mean(ref_words):>12.3f}  (n={len(gen_words)})")
    print(f"{'sentence_exact':<18}{mean(gen_exact):>12.3f}{mean(ref_exact):>12.3f}  (n={len(gen_exact)})")
    print(f"{'sentence_fuzzy':<18}{mean(gen_fuzzy):>12.3f}{mean(ref_fuzzy):>12.3f}  (n={len(gen_fuzzy)})")
    print("\nReference column is the ceiling: the score the ground-truth stories")
    print("themselves achieve. Compare generated against it, not against 1.0.")


if __name__ == "__main__":
    main()
