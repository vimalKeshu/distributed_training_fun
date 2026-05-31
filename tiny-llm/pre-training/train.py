#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model import GPT, GPTConfig
from muon import Muon
from tokenizer_util import load_tokenizer


def log_metrics(
    writer: SummaryWriter,
    metrics_file: Any,
    step: int,
    metrics: dict[str, float],
) -> None:
    """Write a row of metrics to TensorBoard and append it to a JSONL file."""
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)
    metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
    metrics_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small subword-level GPT.")
    parser.add_argument("config", type=Path, help="Path to a JSON training config.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_to_bin(
    text_path: Path,
    target: Path,
    tokenizer: Any,
    *,
    flush_every_tokens: int = 4_000_000,
) -> int:
    """Tokenize a text file to a uint16 token-id stream, one story per document.

    Stories are delimited by an `<|endoftext|>` line (the raw TinyStories format);
    files without that marker are treated as a single document. Each story is
    encoded independently and followed by the `<|endoftext|>` id so the model
    learns document boundaries.

    The file is streamed line by line and the token buffer is flushed to disk
    periodically, so memory stays bounded regardless of corpus size (the full
    1.9 GB train set is ~500M tokens, far too many to hold in a Python list).
    Returns the number of tokens written.
    """
    eot_id = tokenizer.eot_id
    total = 0
    buffer: list[int] = []
    doc_lines: list[str] = []

    with target.open("wb") as out:

        def write_buffer() -> None:
            nonlocal total, buffer
            if buffer:
                np.asarray(buffer, dtype=np.uint16).tofile(out)
                total += len(buffer)
                buffer = []

        def finalize_doc() -> None:
            nonlocal doc_lines
            document = "".join(doc_lines).strip()
            doc_lines = []
            if document:
                buffer.extend(tokenizer.encode(document))
                if eot_id is not None:
                    buffer.append(eot_id)

        with text_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip() == "<|endoftext|>":
                    finalize_doc()
                    if len(buffer) >= flush_every_tokens:
                        write_buffer()
                else:
                    doc_lines.append(line)
        finalize_doc()
        write_buffer()

    return total


def prepare_cache(
    *,
    train_text_path: Path,
    valid_text_path: Path,
    cache_dir: Path,
    tokenizer: Any,
    overwrite: bool,
) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_bin = cache_dir / "train_tokens.bin"
    valid_bin = cache_dir / "valid_tokens.bin"

    for source, target in [(train_text_path, train_bin), (valid_text_path, valid_bin)]:
        if target.exists() and not overwrite:
            continue
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Download data first, for example:\n"
                "python data/download_tinystories.py subset --target-train-tokens 50000000"
            )
        encode_to_bin(source, target, tokenizer)

    manifest = {
        "tokenizer": "bpe",
        "vocab_size": tokenizer.vocab_size,
        "train_text_path": str(train_text_path),
        "valid_text_path": str(valid_text_path),
        "train_bin": str(train_bin),
        "valid_bin": str(valid_bin),
        "train_tokens": int(train_bin.stat().st_size // 2),
        "valid_tokens": int(valid_bin.stat().st_size // 2),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return train_bin, valid_bin


def load_memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    *,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError(f"Dataset has {len(data)} tokens, block_size={block_size} is too large")
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[start : start + block_size].astype(np.int64)) for start in starts]
    )
    y = torch.stack(
        [torch.from_numpy(data[start + 1 : start + 1 + block_size].astype(np.int64)) for start in starts]
    )
    return x.to(device), y.to(device)


def configure_optimizers(model: GPT, training_config: dict[str, Any]) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s).

    With ``optimizer == "muon"`` the 2D hidden weight matrices in the transformer
    blocks are optimized with Muon, while everything else (token embedding / tied
    LM head, RMSNorm gains, biases) stays on AdamW. Otherwise a single AdamW
    handles all parameters.
    """
    weight_decay = training_config["weight_decay"]
    betas = tuple(training_config["betas"])
    optimizer_name = training_config.get("optimizer", "adamw")

    muon_params = []
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_block_matrix = parameter.dim() == 2 and "blocks." in name
        if optimizer_name == "muon" and is_block_matrix:
            muon_params.append(parameter)
        elif parameter.dim() >= 2:
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)

    adamw = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=training_config["learning_rate"],
        betas=betas,
    )

    optimizers: list[torch.optim.Optimizer] = [adamw]
    if optimizer_name == "muon":
        optimizers.insert(
            0,
            Muon(
                muon_params,
                lr=training_config.get("muon_lr", 0.02),
                momentum=training_config.get("muon_momentum", 0.95),
                weight_decay=weight_decay,
            ),
        )

    # Remember each group's base LR so the schedule can scale it by a multiplier.
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def get_lr_multiplier(step: int, training_config: dict[str, Any]) -> float:
    """Cosine schedule expressed as a multiplier in [min_ratio, 1] on each base LR."""
    warmup_steps = training_config["warmup_steps"]
    max_steps = training_config["max_steps"]

    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    if not training_config["lr_decay"]:
        return 1.0
    min_ratio = training_config["min_lr"] / training_config["learning_rate"]
    if step >= max_steps:
        return min_ratio
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_ratio + coeff * (1.0 - min_ratio)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.memmap,
    valid_data: np.memmap,
    training_config: dict[str, Any],
    block_size: int,
    device: str,
    ctx: Any,
) -> dict[str, float]:
    model.eval()
    out = {}
    eval_iters = training_config["eval_iters"]
    batch_size = training_config["batch_size"]

    for split, data in [("train", train_data), ("valid", valid_data)]:
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            x, y = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
            with ctx:
                _, loss = model(x, y)
            losses[index] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    *,
    path: Path,
    model: GPT,
    optimizers: list[torch.optim.Optimizer],
    config: dict[str, Any],
    step: int,
    best_valid_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "model_config": config["model"],
            "config": config,
            "step": step,
            "best_valid_loss": best_valid_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = GPTConfig(**config["model"])
    training_config = config["training"]

    set_seed(config.get("seed", 1337))
    device = select_device(training_config["device"])
    dtype_name = training_config.get("dtype", "float32")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if device == "cpu" or dtype == torch.float32:
        ctx = nullcontext()
    else:
        ctx = torch.autocast(device_type=device, dtype=dtype)

    tokenizer = load_tokenizer(resolve_path(data_config["tokenizer_path"]))
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Config model.vocab_size={model_config.vocab_size} does not match "
            f"tokenizer vocab_size={tokenizer.vocab_size}. Update the config."
        )

    train_bin, valid_bin = prepare_cache(
        train_text_path=resolve_path(data_config["train_text_path"]),
        valid_text_path=resolve_path(data_config["valid_text_path"]),
        cache_dir=resolve_path(data_config["cache_dir"]),
        tokenizer=tokenizer,
        overwrite=data_config["overwrite_cache"],
    )
    train_data = load_memmap(train_bin)
    valid_data = load_memmap(valid_bin)

    out_dir = resolve_path(training_config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-run id so successive runs never overwrite each other's metrics: each
    # gets its own metrics file and TensorBoard subdir (TB shows them as
    # separate, comparable curves). config.json snapshots the run that produced
    # them.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"config_{run_id}.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard" / run_id))
    metrics_file = (out_dir / f"metrics_{run_id}.jsonl").open("w", encoding="utf-8")
    print(f"run_id: {run_id} | metrics: metrics_{run_id}.jsonl | tb: tensorboard/{run_id}")

    raw_model = GPT(model_config).to(device)
    model = raw_model
    if training_config["compile"]:
        # torch.compile uses the Triton inductor backend, which only supports
        # CUDA compute capability >= 7.0. Older cards (e.g. Maxwell/Pascal like
        # the TITAN Xp at 6.1) fall back to eager so training can still run.
        is_cuda = str(device).startswith("cuda")
        capability = torch.cuda.get_device_capability(device) if is_cuda else (0, 0)
        if is_cuda and capability[0] < 7:
            print(
                f"compile requested but GPU compute capability {capability[0]}.{capability[1]} "
                "is too old for Triton (needs >= 7.0); running uncompiled (eager)."
            )
        else:
            model = torch.compile(model)
    optimizers = configure_optimizers(model, training_config)

    print(f"run_name: {config['run_name']}")
    print(f"device: {device}")
    print(f"dtype: {dtype_name}")
    print(f"optimizer: {training_config.get('optimizer', 'adamw')}")
    print(f"vocab_size: {model_config.vocab_size}")
    print(f"parameters: {raw_model.parameter_count() / 1_000_000:.2f}M")
    print(f"train tokens: {len(train_data):,}")
    print(f"valid tokens: {len(valid_data):,}")

    best_valid_loss = float("inf")
    tokens_per_step = (
        training_config["batch_size"]
        * training_config["gradient_accumulation_steps"]
        * model_config.block_size
    )
    start_time = time.time()
    last_log_time = start_time

    for step in range(training_config["max_steps"]):
        lr_mult = get_lr_multiplier(step, training_config)
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * lr_mult

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(training_config["gradient_accumulation_steps"]):
            x, y = get_batch(
                train_data,
                batch_size=training_config["batch_size"],
                block_size=model_config.block_size,
                device=device,
            )
            with ctx:
                _, loss = model(x, y)
                loss = loss / training_config["gradient_accumulation_steps"]
            loss.backward()
            total_loss += loss.item()

        if training_config["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config["grad_clip"])
        for optimizer in optimizers:
            optimizer.step()

        if step % training_config["log_interval"] == 0:
            now = time.time()
            elapsed = now - last_log_time
            steps_since_log = training_config["log_interval"] if step > 0 else 1
            tok_per_sec = (tokens_per_step * steps_since_log) / max(elapsed, 1e-9)
            last_log_time = now
            adamw_lr = optimizers[-1].param_groups[0]["lr"]
            print(
                f"step {step:05d} | loss {total_loss:.4f} | "
                f"lr {adamw_lr:.2e} | {tok_per_sec:,.0f} tok/s"
            )
            log_metrics(
                writer,
                metrics_file,
                step,
                {"train/loss": total_loss, "train/lr": adamw_lr, "perf/tokens_per_sec": tok_per_sec},
            )

        if step % training_config["eval_interval"] == 0 or step == training_config["max_steps"] - 1:
            losses = estimate_loss(
                model,
                train_data,
                valid_data,
                training_config,
                model_config.block_size,
                device,
                ctx,
            )
            print(
                f"eval step {step:05d} | train {losses['train']:.4f} | "
                f"valid {losses['valid']:.4f}"
            )
            log_metrics(
                writer,
                metrics_file,
                step,
                {"eval/train_loss": losses["train"], "eval/valid_loss": losses["valid"]},
            )
            if losses["valid"] < best_valid_loss:
                best_valid_loss = losses["valid"]
                save_checkpoint(
                    path=out_dir / "best.pt",
                    model=raw_model,
                    optimizers=optimizers,
                    config=config,
                    step=step,
                    best_valid_loss=best_valid_loss,
                )

        if step > 0 and step % training_config["save_interval"] == 0:
            save_checkpoint(
                path=out_dir / f"step_{step:06d}.pt",
                model=raw_model,
                optimizers=optimizers,
                config=config,
                step=step,
                best_valid_loss=best_valid_loss,
            )

    save_checkpoint(
        path=out_dir / "last.pt",
        model=raw_model,
        optimizers=optimizers,
        config=config,
        step=training_config["max_steps"],
        best_valid_loss=best_valid_loss,
    )
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"done in {elapsed_minutes:.1f} min | best_valid_loss {best_valid_loss:.4f}")

    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak cuda memory: {peak_gb:.2f} GB")
        writer.add_scalar("perf/peak_cuda_gb", peak_gb, training_config["max_steps"])

    writer.close()
    metrics_file.close()


if __name__ == "__main__":
    main()
