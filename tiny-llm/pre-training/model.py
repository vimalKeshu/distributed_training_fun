from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    position_encoding: str = "rope"
    sliding_window: int | None = None
    qk_norm: bool = False
    grad_checkpoint: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.dropout(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def alibi_slopes(n_head: int) -> torch.Tensor:
    def slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_head).is_integer():
        slopes = slopes_power_of_2(n_head)
    else:
        closest_power = 2 ** math.floor(math.log2(n_head))
        slopes = slopes_power_of_2(closest_power)
        extra = alibi_slopes(2 * closest_power)[0::2][: n_head - closest_power]
        slopes.extend(extra.tolist())
    return torch.tensor(slopes, dtype=torch.float32)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if config.position_encoding == "rope" and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if config.qk_norm:
            self.q_norm: RMSNorm | None = RMSNorm(self.head_dim)
            self.k_norm: RMSNorm | None = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        if config.sliding_window is not None:
            positions = torch.arange(config.block_size)
            distance = positions[:, None] - positions[None, :]
            causal_mask &= distance <= config.sliding_window
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size))

        if config.position_encoding == "alibi":
            slopes = alibi_slopes(config.n_head).view(1, config.n_head, 1, 1)
            positions = torch.arange(config.block_size)
            distance = positions[:, None] - positions[None, :]
            bias = -slopes * distance.view(1, 1, config.block_size, config.block_size)
            self.register_buffer("alibi_bias", bias)
        else:
            self.alibi_bias = None

        if config.position_encoding == "rope":
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            positions = torch.arange(config.block_size, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            rope = torch.repeat_interleave(freqs, repeats=2, dim=-1)
            self.register_buffer("rope_cos", rope.cos().view(1, 1, config.block_size, self.head_dim))
            self.register_buffer("rope_sin", rope.sin().view(1, 1, config.block_size, self.head_dim))
        else:
            self.rope_cos = None
            self.rope_sin = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embd = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.config.position_encoding == "rope":
            cos = self.rope_cos[:, :, :seq_len, :]
            sin = self.rope_sin[:, :, :seq_len, :]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        dropout_p = self.config.dropout if self.training else 0.0
        if self.alibi_bias is None and self.config.sliding_window is None:
            # Fast path: plain causal attention via fused/memory-efficient SDPA.
            # This avoids materializing the B x H x T x T score matrix, which is
            # what makes longer context (e.g. 1024) fit on a 12GB card.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        else:
            # Fold the causal/sliding-window mask and optional ALiBi bias into a
            # single additive attn_mask, then let SDPA handle the softmax.
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            bias = torch.zeros(1, self.n_head, seq_len, seq_len, dtype=q.dtype, device=q.device)
            if self.alibi_bias is not None:
                bias = bias + self.alibi_bias[:, :, :seq_len, :seq_len].to(q.dtype)
            bias = bias.masked_fill(~mask, float("-inf"))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=dropout_p)

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if config.position_encoding == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.position_embedding = None
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # GPT-2 style scaled init for the residual output projections: each block
        # adds two residual contributions (attention + MLP), so scaling these by
        # 1/sqrt(2 * n_layer) keeps the residual stream variance in check as depth
        # grows, which stabilizes training and lowers loss.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")

        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=idx.device)
            x = x + self.position_embedding(positions)[None, :, :]

        x = self.dropout(x)
        for block in self.blocks:
            if self.config.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

