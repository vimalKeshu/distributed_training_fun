"""
~1.3B decoder-only transformer (LLaMA-style).

Uses PyTorch native modules throughout:
  - nn.TransformerEncoderLayer with causal mask
  - nn.RMSNorm (PyTorch 2.4+)
  - F.scaled_dot_product_attention (FlashAttention backend)
  - nn.SiLU for activation

Config (~1.3B params):
  - 24 layers, 2048 hidden, 16 heads, 5504 FFN, 50257 vocab (GPT-2)

Usage:
  from model_1b import Decoder1B
  model = Decoder1B()
  logits = model(input_ids)  # (B, S) → (B, S, V)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257      # GPT-2 tokenizer
    hidden_size: int = 2048
    intermediate_size: int = 5504  # SwiGLU: ~2.7x hidden
    num_layers: int = 24
    num_heads: int = 16
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    bias: bool = False


# -- RoPE via PyTorch ops only ----------------------------------------------

def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0,
                     device=None, dtype=torch.float32):
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    dim = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    freqs = torch.outer(pos, 1.0 / (theta ** (dim / head_dim)))  # (S, D/2)
    cos, sin = freqs.cos(), freqs.sin()
    return cos, sin  # each (S, head_dim//2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (B, H, S, D)
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# -- SwiGLU FFN using nn.Linear + nn.SiLU ----------------------------------

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.bias)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.bias)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


# -- Attention using F.scaled_dot_product_attention -------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.bias)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.bias)
        self.attn_drop = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, S, D)

        q = apply_rope(q, cos[:S], sin[:S])
        k = apply_rope(k, cos[:S], sin[:S])

        # PyTorch native SDPA — uses FlashAttention/memory-efficient backend
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.resid_drop(self.out(out))


# -- Transformer Block with pre-norm (RMSNorm) -----------------------------

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# -- Full Model -------------------------------------------------------------

class Decoder1B(nn.Module):
    def __init__(self, **overrides):
        super().__init__()
        self.cfg = ModelConfig(**overrides)
        c = self.cfg

        self.tok_emb = nn.Embedding(c.vocab_size, c.hidden_size)
        self.drop = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

        # Tie embedding weights with lm_head
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE cache
        head_dim = c.hidden_size // c.num_heads
        cos, sin = build_rope_cache(c.max_seq_len, head_dim, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        self._report()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _report(self):
        total = sum(p.numel() for p in self.parameters())
        # lm_head shares tok_emb — subtract once
        unique = total - self.lm_head.weight.numel()
        print(f"Decoder1B: {unique/1e9:.2f}B unique params ({total/1e9:.2f}B with tied head)")
        c = self.cfg
        print(f"  layers={c.num_layers}, hidden={c.hidden_size}, heads={c.num_heads}, "
              f"ffn={c.intermediate_size}, vocab={c.vocab_size}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, S) → (B, S, vocab_size)"""
        x = self.drop(self.tok_emb(input_ids))
        cos, sin = self.rope_cos, self.rope_sin

        for layer in self.layers:
            x = layer(x, cos, sin)

        return self.lm_head(self.norm(x))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = Decoder1B().to(device=device, dtype=dtype)

    ids = torch.randint(0, 50257, (2, 512), device=device)
    with torch.no_grad():
        logits = model(ids)
    print(f"Output: {logits.shape}")  # (2, 512, 50257)
    if device == "cuda":
        print(f"GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")