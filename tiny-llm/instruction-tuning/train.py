#!/usr/bin/env python3
"""Instruction fine-tuning (SFT) of the GPT on TinyStories-Instruct.

This loads a pre-trained checkpoint produced by the `pre-training/` stage and
continues training on instruction-formatted examples, masking the loss on the
prompt so the model learns to generate the story given the instruction fields.

It uses the same subword tokenizer, Muon/AdamW optimizer split, and cosine LR
schedule as pre-training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data_sft import (
    index_stream,
    load_or_build_examples,
    make_batch,
)
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
    parser = argparse.ArgumentParser(description="Instruction fine-tune a subword GPT.")
    parser.add_argument("config", type=Path, help="Path to a JSON SFT config.")
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


def load_pretrained_weights(model: GPT, checkpoint: dict[str, Any]) -> None:
    """Load matching tensors from a pre-training checkpoint.

    Buffers such as `rope_cos`/`causal_mask` are sized by `block_size`; if the
    fine-tuning block_size differs they are skipped and keep their freshly
    computed values. All learned parameters transfer regardless of block_size.
    """
    state = checkpoint["model"]
    own = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in state.items():
        if key in own and own[key].shape == value.shape:
            loaded[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(loaded, strict=False)
    print(f"loaded {len(loaded)} tensors from pretrained checkpoint")
    if skipped:
        print(f"  skipped (shape mismatch / position buffers): {skipped}")


def configure_optimizers(model: GPT, training_config: dict[str, Any]) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s); mirrors pre-training (Muon on 2D block weights)."""
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
    examples: dict[str, Any],
    samplers: dict[str, Any],
    training_config: dict[str, Any],
    block_size: int,
    device: str,
    ctx: Any,
) -> dict[str, float]:
    model.eval()
    out = {}
    eval_iters = training_config["eval_iters"]
    for split in ("train", "valid"):
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            batch_indices = next(samplers[split])
            x, y = make_batch(examples[split], batch_indices, block_size=block_size, device=device)
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
    model_config: GPTConfig,
    step: int,
    best_valid_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "model_config": vars(model_config),
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
    training_config = config["training"]

    set_seed(config.get("seed", 1337))
    device = select_device(training_config["device"])
    dtype_name = training_config.get("dtype", "float32")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if device == "cpu" or dtype == torch.float32:
        ctx = nullcontext()
    else:
        ctx = torch.autocast(device_type=device, dtype=dtype)

    # The architecture is fixed by the pretrained checkpoint; block_size may be
    # overridden in the config (RoPE/causal buffers are rebuilt accordingly).
    init_from = resolve_path(training_config["init_from"])
    checkpoint = torch.load(init_from, map_location="cpu")
    # Architecture is inherited from the checkpoint; any field present in the
    # config's "model" block overrides it (e.g. block_size for a longer context,
    # or dropout to regularize fine-tuning on a small dataset).
    model_args = dict(checkpoint["model_config"])
    model_args.update(config.get("model", {}))
    model_config = GPTConfig(**model_args)

    tokenizer = load_tokenizer(resolve_path(data_config["tokenizer_path"]))
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not match the "
            f"pretrained model vocab_size={model_config.vocab_size}. Use the same "
            "tokenizer that the base model was trained with."
        )

    cache_dir = resolve_path(data_config["cache_dir"])
    require_marker = data_config.get("require_story_marker", True)
    overwrite = data_config.get("overwrite_cache", False)
    # Cap the (multi-GB) train corpus; valid is small so it is used in full.
    max_train_tokens = data_config.get("max_train_tokens")
    examples = {
        "train": load_or_build_examples(
            text_path=resolve_path(data_config["train_text_path"]),
            cache_path=cache_dir / "train_sft.pkl",
            tokenizer=tokenizer,
            block_size=model_config.block_size,
            require_story_marker=require_marker,
            overwrite=overwrite,
            max_tokens=max_train_tokens,
        ),
        "valid": load_or_build_examples(
            text_path=resolve_path(data_config["valid_text_path"]),
            cache_path=cache_dir / "valid_sft.pkl",
            tokenizer=tokenizer,
            block_size=model_config.block_size,
            require_story_marker=require_marker,
            overwrite=overwrite,
        ),
    }

    seed = config.get("seed", 1337)
    samplers = {
        "train": index_stream(len(examples["train"]), training_config["batch_size"], seed),
        "valid": index_stream(len(examples["valid"]), training_config["batch_size"], seed + 1),
    }

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
    load_pretrained_weights(raw_model, checkpoint)
    model = raw_model
    if training_config["compile"]:
        # torch.compile's Triton backend needs CUDA compute capability >= 7.0;
        # older cards (e.g. TITAN Xp at 6.1) fall back to eager.
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
    print(f"init_from: {init_from}")
    print(f"device: {device} | dtype: {dtype_name}")
    print(f"optimizer: {training_config.get('optimizer', 'adamw')} | vocab_size: {model_config.vocab_size}")
    print(f"parameters: {raw_model.parameter_count() / 1_000_000:.2f}M")
    print(f"train examples: {len(examples['train']):,} | valid examples: {len(examples['valid']):,}")

    best_valid_loss = float("inf")
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
            batch_indices = next(samplers["train"])
            x, y = make_batch(
                examples["train"], batch_indices, block_size=model_config.block_size, device=device
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
            steps_per_sec = steps_since_log / max(elapsed, 1e-9)
            last_log_time = now
            adamw_lr = optimizers[-1].param_groups[0]["lr"]
            print(
                f"step {step:05d} | loss {total_loss:.4f} | "
                f"lr {adamw_lr:.2e} | {steps_per_sec:.2f} steps/s"
            )
            log_metrics(
                writer,
                metrics_file,
                step,
                {"train/loss": total_loss, "train/lr": adamw_lr, "perf/steps_per_sec": steps_per_sec},
            )

        if step % training_config["eval_interval"] == 0 or step == training_config["max_steps"] - 1:
            losses = estimate_loss(
                model, examples, samplers, training_config, model_config.block_size, device, ctx
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
                    model_config=model_config,
                    step=step,
                    best_valid_loss=best_valid_loss,
                )

        if step > 0 and step % training_config["save_interval"] == 0:
            save_checkpoint(
                path=out_dir / f"step_{step:06d}.pt",
                model=raw_model,
                optimizers=optimizers,
                config=config,
                model_config=model_config,
                step=step,
                best_valid_loss=best_valid_loss,
            )

    save_checkpoint(
        path=out_dir / "last.pt",
        model=raw_model,
        optimizers=optimizers,
        config=config,
        model_config=model_config,
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
