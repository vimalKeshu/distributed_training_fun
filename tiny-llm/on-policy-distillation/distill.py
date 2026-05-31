#!/usr/bin/env python3
"""On-policy distillation of an instruction-following GPT.

The idea (vs. SFT): instead of training the student on fixed ground-truth
stories with a one-hot target, we let the **student generate its own response**
to each prompt (on-policy rollout) and train it to match the **teacher's full
next-token distribution** on those self-generated tokens. The per-token KL to
the teacher acts as a dense reward at every position — combining the on-policy
states of RL with the dense supervision of SFT.

Pipeline per optimizer step:
  1. sample a batch of instruction prompts (text up to ``Story:``)
  2. the student rolls out a continuation for each (sampling, no grad)
  3. run teacher (frozen) and student over ``prompt + response`` teacher-forced
  4. compute per-token KL(student || teacher) over the *response* tokens only
  5. backprop into the student

The student starts from the pretrained base; the teacher is the SFT instruct
model. Both must share the exact same tokenizer/vocab (the KL aligns them
token-by-token). The student is saved in the same checkpoint format as the other
stages, so ``instruction-tuning/sample.py`` and ``eval_following.py`` work on it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_prompts import index_stream, load_or_build_prompts
from model import GPT, GPTConfig
from muon import Muon
from tokenizer_util import load_tokenizer


# --------------------------------------------------------------------------- #
# Small utilities (mirrors the other training stages)
# --------------------------------------------------------------------------- #
def log_metrics(writer: SummaryWriter, metrics_file: Any, step: int, metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)
    metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
    metrics_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy distill an instruction GPT.")
    parser.add_argument("config", type=Path, help="Path to a JSON distillation config.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(checkpoint: dict[str, Any], device: str, overrides: dict[str, Any] | None = None) -> tuple[GPT, GPTConfig]:
    """Build a GPT from a checkpoint, optionally overriding model-config fields."""
    model_args = dict(checkpoint["model_config"])
    if overrides:
        model_args.update(overrides)
    model_config = GPTConfig(**model_args)
    model = GPT(model_config).to(device)
    # Partial load: buffers sized by block_size (rope/causal mask) are skipped if
    # the override changed block_size; all learned parameters still transfer.
    own = model.state_dict()
    loaded = {k: v for k, v in checkpoint["model"].items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(loaded, strict=False)
    return model, model_config


def configure_optimizers(model: GPT, training_config: dict[str, Any]) -> list[torch.optim.Optimizer]:
    """Student optimizer(s); mirrors the other stages (Muon on 2D block weights)."""
    weight_decay = training_config["weight_decay"]
    betas = tuple(training_config["betas"])
    optimizer_name = training_config.get("optimizer", "adamw")

    muon_params, decay_params, no_decay_params = [], [], []
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


# --------------------------------------------------------------------------- #
# Rollout + KL loss
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout(
    student: GPT,
    prompt_ids: np.ndarray,
    *,
    block_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eot_id: int | None,
    device: str,
) -> tuple[torch.Tensor, int, int]:
    """Sample one student continuation for a prompt.

    Returns ``(full_seq, prompt_len, true_len)`` where ``full_seq`` is
    ``prompt + response`` (truncated at the first generated EOT) and
    ``true_len`` is its length. Generation runs in eval mode (no dropout).
    """
    prompt_len = int(len(prompt_ids))
    eff_max = min(max_new_tokens, block_size - prompt_len)
    idx = torch.from_numpy(prompt_ids.astype(np.int64))[None, :].to(device)
    out = student.generate(idx, max_new_tokens=eff_max, temperature=temperature, top_k=top_k)
    seq = out[0]  # [prompt_len + eff_max]

    true_len = int(seq.size(0))
    if eot_id is not None:
        response = seq[prompt_len:]
        hits = (response == eot_id).nonzero(as_tuple=False)
        if hits.numel() > 0:
            # Keep through the first EOT so the student learns to stop there.
            true_len = prompt_len + int(hits[0].item()) + 1
    return seq[:true_len], prompt_len, true_len


def build_rollout_batch(
    seqs: list[torch.Tensor],
    prompt_lens: list[int],
    true_lens: list[int],
    *,
    pad_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad rollouts into ``x`` and a boolean mask over response positions.

    ``mask[b, t]`` marks the positions whose *prediction target* (token t+1) is a
    response token of sequence b — i.e. ``prompt_len-1 <= t <= true_len-2``. The
    prompt and the padding are excluded, exactly like SFT's prompt masking.
    """
    batch = len(seqs)
    max_len = max(true_lens)
    x = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
    for row, seq in enumerate(seqs):
        x[row, : seq.size(0)] = seq

    positions = torch.arange(max_len, device=device)
    plen = torch.tensor(prompt_lens, device=device)[:, None]
    tlen = torch.tensor(true_lens, device=device)[:, None]
    mask = (positions[None, :] >= plen - 1) & (positions[None, :] <= tlen - 2)
    return x, mask


def distill_loss(
    student: GPT,
    teacher: GPT,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    kl_type: str,
    temperature: float,
) -> torch.Tensor:
    """Per-token KL between teacher and student over the masked (response) positions.

    reverse KL = KL(student || teacher) (mode-seeking, the on-policy default);
    forward KL = KL(teacher || student) (mode-covering, classic distillation).
    Teacher logits carry no gradient, so only the student is trained.
    """
    with torch.no_grad():
        teacher_logits, _ = teacher(x)
    student_logits, _ = student(x)

    log_t = F.log_softmax(teacher_logits / temperature, dim=-1)  # detached (teacher under no_grad)
    log_s = F.log_softmax(student_logits / temperature, dim=-1)

    if kl_type == "reverse":
        probs = log_s.exp()
        kl = (probs * (log_s - log_t)).sum(dim=-1)
    elif kl_type == "forward":
        probs = log_t.exp()
        kl = (probs * (log_t - log_s)).sum(dim=-1)
    else:
        raise ValueError(f"kl_type must be 'reverse' or 'forward', got {kl_type!r}")

    mask_f = mask.to(kl.dtype)
    return (kl * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def run_rollouts(
    student: GPT,
    prompts: Any,
    indices: np.ndarray,
    *,
    block_size: int,
    distill_config: dict[str, Any],
    eot_id: int | None,
    pad_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rollouts for a batch of prompt indices and pack them for the loss."""
    student.eval()
    seqs, plens, tlens = [], [], []
    for i in indices:
        seq, plen, tlen = rollout(
            student,
            prompts.ids[i],
            block_size=block_size,
            max_new_tokens=distill_config["max_new_tokens"],
            temperature=distill_config.get("rollout_temperature", 1.0),
            top_k=distill_config.get("rollout_top_k"),
            eot_id=eot_id,
            device=device,
        )
        seqs.append(seq)
        plens.append(plen)
        tlens.append(tlen)
    student.train()
    return build_rollout_batch(seqs, plens, tlens, pad_id=pad_id, device=device)


@torch.no_grad()
def estimate_valid_kl(
    student: GPT,
    teacher: GPT,
    prompts: Any,
    sampler: Any,
    *,
    block_size: int,
    distill_config: dict[str, Any],
    eot_id: int | None,
    pad_id: int,
    device: str,
    eval_iters: int,
) -> float:
    """Mean per-token KL on held-out prompts (student rollouts vs. teacher)."""
    student.eval()
    total = 0.0
    for _ in range(eval_iters):
        indices = next(sampler)
        x, mask = run_rollouts(
            student, prompts, indices,
            block_size=block_size, distill_config=distill_config,
            eot_id=eot_id, pad_id=pad_id, device=device,
        )
        total += distill_loss(
            student, teacher, x, mask,
            kl_type=distill_config.get("kl_type", "reverse"),
            temperature=distill_config.get("kl_temperature", 1.0),
        ).item()
    student.train()
    return total / max(1, eval_iters)


def save_checkpoint(*, path: Path, model: GPT, config: dict[str, Any], model_config: GPTConfig, step: int, best_valid_kl: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": vars(model_config),
            "config": config,
            "step": step,
            "best_valid_kl": best_valid_kl,
        },
        path,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    distill_config = config["distill"]
    training_config = config["training"]

    set_seed(config.get("seed", 1337))
    device = select_device(training_config["device"])

    tokenizer = load_tokenizer(resolve_path(data_config["tokenizer_path"]))
    eot_id = tokenizer.eot_id
    pad_id = eot_id if eot_id is not None else 0

    # ---- teacher (frozen SFT model) and student (starts from base) ----------
    # weights_only=False: these are our own trusted checkpoints (they store a
    # GPTConfig object, not just tensors).
    teacher_ckpt = torch.load(
        resolve_path(distill_config["teacher_from"]), map_location="cpu", weights_only=False
    )
    teacher, teacher_config = load_model(teacher_ckpt, device)
    teacher.eval()
    teacher.requires_grad_(False)

    student_ckpt = torch.load(
        resolve_path(distill_config["student_init_from"]), map_location="cpu", weights_only=False
    )
    student, student_config = load_model(student_ckpt, device, overrides=config.get("model", {}))

    for label, cfg in (("teacher", teacher_config), ("student", student_config)):
        if tokenizer.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"{label} vocab_size={cfg.vocab_size} != tokenizer vocab_size="
                f"{tokenizer.vocab_size}; teacher, student and tokenizer must match."
            )
    if teacher_config.vocab_size != student_config.vocab_size:
        raise ValueError("Teacher and student must share the same vocab for token-aligned KL.")

    block_size = student_config.block_size

    # ---- prompts ------------------------------------------------------------
    cache_dir = resolve_path(data_config["cache_dir"])
    overwrite = data_config.get("overwrite_cache", False)
    min_response = distill_config.get("min_response_tokens", 64)
    prompts = {
        "train": load_or_build_prompts(
            text_path=resolve_path(data_config["train_text_path"]),
            cache_path=cache_dir / "train_prompts.pkl",
            tokenizer=tokenizer,
            block_size=block_size,
            min_response_tokens=min_response,
            overwrite=overwrite,
            max_tokens=data_config.get("max_train_tokens"),
        ),
        "valid": load_or_build_prompts(
            text_path=resolve_path(data_config["valid_text_path"]),
            cache_path=cache_dir / "valid_prompts.pkl",
            tokenizer=tokenizer,
            block_size=block_size,
            min_response_tokens=min_response,
            overwrite=overwrite,
        ),
    }

    seed = config.get("seed", 1337)
    samplers = {
        "train": index_stream(len(prompts["train"]), training_config["batch_size"], seed),
        "valid": index_stream(len(prompts["valid"]), training_config["batch_size"], seed + 1),
    }

    optimizers = configure_optimizers(student, training_config)

    out_dir = resolve_path(training_config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"config_{run_id}.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard" / run_id))
    metrics_file = (out_dir / f"metrics_{run_id}.jsonl").open("w", encoding="utf-8")

    print(f"run_name: {config['run_name']} | run_id: {run_id}")
    print(f"teacher: {distill_config['teacher_from']}")
    print(f"student init: {distill_config['student_init_from']}")
    print(f"device: {device} | kl: {distill_config.get('kl_type', 'reverse')} | "
          f"rollout T={distill_config.get('rollout_temperature', 1.0)} | "
          f"max_new_tokens={distill_config['max_new_tokens']}")
    print(f"parameters (student): {student.parameter_count() / 1_000_000:.2f}M")
    print(f"train prompts: {len(prompts['train']):,} | valid prompts: {len(prompts['valid']):,}")

    kl_type = distill_config.get("kl_type", "reverse")
    kl_temp = distill_config.get("kl_temperature", 1.0)
    accum = training_config["gradient_accumulation_steps"]

    best_valid_kl = float("inf")
    start_time = time.time()
    last_log_time = start_time

    for step in range(training_config["max_steps"]):
        lr_mult = get_lr_multiplier(step, training_config)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lr_mult
            optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(accum):
            indices = next(samplers["train"])
            x, mask = run_rollouts(
                student, prompts["train"], indices,
                block_size=block_size, distill_config=distill_config,
                eot_id=eot_id, pad_id=pad_id, device=device,
            )
            loss = distill_loss(student, teacher, x, mask, kl_type=kl_type, temperature=kl_temp) / accum
            loss.backward()
            total_loss += loss.item()

        if training_config["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), training_config["grad_clip"])
        for optimizer in optimizers:
            optimizer.step()

        if step % training_config["log_interval"] == 0:
            now = time.time()
            steps_since = training_config["log_interval"] if step > 0 else 1
            steps_per_sec = steps_since / max(now - last_log_time, 1e-9)
            last_log_time = now
            lr_now = optimizers[-1].param_groups[0]["lr"]
            print(f"step {step:05d} | kl {total_loss:.4f} | lr {lr_now:.2e} | {steps_per_sec:.2f} steps/s")
            log_metrics(writer, metrics_file, step,
                        {"train/kl": total_loss, "train/lr": lr_now, "perf/steps_per_sec": steps_per_sec})

        if step % training_config["eval_interval"] == 0 or step == training_config["max_steps"] - 1:
            valid_kl = estimate_valid_kl(
                student, teacher, prompts["valid"], samplers["valid"],
                block_size=block_size, distill_config=distill_config,
                eot_id=eot_id, pad_id=pad_id, device=device,
                eval_iters=training_config["eval_iters"],
            )
            print(f"eval step {step:05d} | valid kl {valid_kl:.4f}")
            log_metrics(writer, metrics_file, step, {"eval/valid_kl": valid_kl})
            if valid_kl < best_valid_kl:
                best_valid_kl = valid_kl
                save_checkpoint(path=out_dir / "best.pt", model=student, config=config,
                                model_config=student_config, step=step, best_valid_kl=best_valid_kl)

        if step > 0 and step % training_config["save_interval"] == 0:
            save_checkpoint(path=out_dir / f"step_{step:06d}.pt", model=student, config=config,
                            model_config=student_config, step=step, best_valid_kl=best_valid_kl)

    save_checkpoint(path=out_dir / "last.pt", model=student, config=config,
                    model_config=student_config, step=training_config["max_steps"], best_valid_kl=best_valid_kl)
    print(f"done in {(time.time() - start_time) / 60:.1f} min | best_valid_kl {best_valid_kl:.4f}")
    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak cuda memory: {peak_gb:.2f} GB")
    writer.close()
    metrics_file.close()


if __name__ == "__main__":
    main()
