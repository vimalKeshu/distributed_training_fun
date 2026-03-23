"""
Usage:
  torchrun --nproc_per_node=8 trainer.py --config dataloader_config.yaml \
      --optimizer adamw --dtype bf16 --run-name adamw_bf16

  torchrun --nproc_per_node=8 trainer.py --config dataloader_config.yaml \
      --optimizer flash_adamw --dtype bf16 --run-name flash_bf16

  python compare_runs.py --runs adamw_bf16 flash_bf16
"""

import argparse
import json
import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed_dataloader import DistributedParquetDataloader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Toy Transformer LM
# ---------------------------------------------------------------------------
# from flashoptim.models.toy_model_1 import ToyTransformerLM
from flashoptim.models.toy_model_2 import Decoder1B


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.01,
                    master_weight_bits: int = 24):
    name = name.lower().replace("-", "_")
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "flash_adamw":
        from flashoptim import FlashAdamW
        return FlashAdamW(params, lr=lr, weight_decay=weight_decay,
                          master_weight_bits=master_weight_bits)
    elif name == "flash_adam":
        from flashoptim import FlashAdam
        return FlashAdam(params, lr=lr, weight_decay=weight_decay)
    elif name == "flash_lion":
        from flashoptim import FlashLion
        return FlashLion(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    def __init__(self, run_name: str, rank: int, log_dir: str = "runs"):
        self.run_name, self.rank, self.log_dir = run_name, rank, log_dir
        self.steps = []
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, step, loss, lr, step_time, tokens_per_sec, gpu_mem_mb):
        self.steps.append({
            "step": step,
            "loss": round(loss, 6),
            "perplexity": round(math.exp(min(loss, 20)), 4),
            "lr": lr,
            "step_time_s": round(step_time, 4),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "gpu_mem_allocated_mb": round(gpu_mem_mb, 1),
        })

    def save(self):
        if self.rank != 0:
            return
        path = os.path.join(self.log_dir, f"{self.run_name}.json")
        data = {
            "run_name": self.run_name,
            "total_steps": len(self.steps),
            "final_loss": self.steps[-1]["loss"] if self.steps else None,
            "final_perplexity": self.steps[-1]["perplexity"] if self.steps else None,
            "avg_step_time": round(
                sum(s["step_time_s"] for s in self.steps) / max(len(self.steps), 1), 4),
            "avg_tokens_per_sec": round(
                sum(s["tokens_per_sec"] for s in self.steps) / max(len(self.steps), 1), 1),
            "peak_gpu_mem_mb": round(
                max((s["gpu_mem_allocated_mb"] for s in self.steps), default=0), 1),
            "steps": self.steps,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Metrics saved → {path}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(config_path: str, steps: int, lr: float, ckpt_every: int,
          optimizer_name: str, run_name: str, master_weight_bits: int,
          log_dir: str, dtype: str):

    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    train_dtype = dtype_map.get(dtype, torch.float32)
    is_flash = optimizer_name.startswith("flash")

    # --- Dataloader ----------------------------------------------------------
    loader = DistributedParquetDataloader(
        config_path=config_path, rank=rank, world_size=world_size, device=device
    )
    loader.start()
    seq_length = loader.seq_length
    batch_size = loader.batch_size
    tokens_per_batch = seq_length * batch_size * world_size

    # =========================================================================
    # Model + Optimizer setup — two distinct paths
    # =========================================================================

    # model = ToyTransformerLM(max_seq_len=seq_length).to(device)  # fp32 on correct GPU
    model = Decoder1B().to(device)

    if is_flash:
        # ---- FlashOptim path ------------------------------------------------
        # 1. Cast model to bf16/fp16 using flashoptim's utility
        # 2. FlashOptim handles master weights internally
        # 3. No autocast needed — forward/backward run in native bf16
        # 4. No GradScaler needed for bf16
        if train_dtype != torch.float32:
            from flashoptim import cast_model
            cast_model(model, dtype=train_dtype)

        if distributed:
            model = DDP(model, device_ids=[local_rank])

        optimizer = build_optimizer(
            optimizer_name, model.parameters(), lr=lr,
            master_weight_bits=master_weight_bits,
        )

        use_autocast = False
        use_scaler = False
        scaler = None
        model_dtype_str = str(train_dtype).split(".")[-1]

    else:
        # ---- Standard optimizer path ----------------------------------------
        # 1. Model stays fp32
        # 2. torch.amp.autocast handles bf16/fp16 for forward + loss
        # 3. GradScaler needed for fp16 only (bf16 doesn't need it)
        # 4. Optimizer updates in fp32 (proper mixed precision)
        if distributed:
            model = DDP(model, device_ids=[local_rank])

        optimizer = build_optimizer(
            optimizer_name, model.parameters(), lr=lr,
            master_weight_bits=master_weight_bits,
        )

        use_autocast = (train_dtype != torch.float32)
        use_scaler = (train_dtype == torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
        model_dtype_str = "fp32 (autocast→" + str(train_dtype).split(".")[-1] + ")" \
            if use_autocast else "fp32"

    loss_fn = nn.CrossEntropyLoss()

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"  Run:       {run_name}")
        print(f"  Optimizer: {optimizer_name}")
        if is_flash:
            print(f"  Master wt: {master_weight_bits}-bit")
        print(f"  Model:     {model_dtype_str}")
        print(f"  Autocast:  {use_autocast}  |  GradScaler: {use_scaler}")
        print(f"  Params:    {param_count:,}")
        print(f"  Seq len:   {seq_length}, Batch: {batch_size}, World: {world_size}")
        print(f"  Steps:     {steps}, LR: {lr}")
        print(f"{'='*60}\n")

    # --- Metrics -------------------------------------------------------------
    metrics = MetricsLogger(run_name, rank, log_dir=log_dir)

    # =========================================================================
    # Training loop — two paths for forward/backward
    # =========================================================================

    model.train()
    for step in range(1, steps + 1):
        t0 = time.monotonic()

        batch = loader.get_batch()
        inputs, targets = batch[:, :-1], batch[:, 1:]

        if is_flash:
            # FlashOptim: native bf16 forward, no autocast
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif use_scaler:
            # Standard fp16: autocast + GradScaler
            with torch.amp.autocast("cuda", dtype=train_dtype):
                logits = model(inputs)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        elif use_autocast:
            # Standard bf16: autocast only (no scaler needed)
            with torch.amp.autocast("cuda", dtype=train_dtype):
                logits = model(inputs)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            # Pure fp32
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step_time = time.monotonic() - t0
        tps = tokens_per_batch / step_time
        gpu_mem = torch.cuda.memory_allocated(device) / 1e6

        metrics.log(step=step, loss=loss.item(), lr=lr,
                    step_time=step_time, tokens_per_sec=tps, gpu_mem_mb=gpu_mem)

        if rank == 0 and (step % 10 == 0 or step == 1):
            ppl = math.exp(min(loss.item(), 20))
            print(
                f"step {step:>5d} | loss {loss.item():.4f} | ppl {ppl:>8.2f} | "
                f"{tps:,.0f} tok/s | mem {gpu_mem:,.0f} MB | {step_time:.3f}s"
            )

        if ckpt_every and step % ckpt_every == 0:
            loader.save_state(f"{log_dir}/{run_name}_state_r{rank}_s{step}.json")

    loader.stop()
    metrics.save()

    if rank == 0:
        print(f"\nDone. Dataloader metrics: {loader.get_metrics()}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="dataloader_config.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ckpt-every", type=int, default=0, help="0 = disabled")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "flash_adamw", "flash_adam", "flash_lion"])
    parser.add_argument("--master-weight-bits", type=int, default=24,
                        help="FlashOptim master weight precision (16 or 24)")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--log-dir", type=str, default="runs")
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"{args.optimizer}_{args.dtype}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")

    train(
        args.config, args.steps, args.lr, args.ckpt_every,
        args.optimizer, args.run_name, args.master_weight_bits,
        args.log_dir, args.dtype,
    )

    if dist.is_initialized():
        dist.destroy_process_group()