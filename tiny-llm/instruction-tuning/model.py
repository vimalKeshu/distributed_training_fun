"""Re-export the GPT model from the pre-training package.

Instruction fine-tuning shares the exact same byte-level architecture as
pre-training, so rather than copy `model.py` we load it from the sibling
`pre-training/` directory. This keeps a single source of truth: any change to
the model definition automatically applies to both stages.

We load it by file path (under a distinct module name) to avoid colliding with
this file, which is itself named `model`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PRETRAIN_MODEL = Path(__file__).resolve().parent.parent / "pre-training" / "model.py"
_spec = importlib.util.spec_from_file_location("pretrain_model", _PRETRAIN_MODEL)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load pre-training model from {_PRETRAIN_MODEL}")
_module = importlib.util.module_from_spec(_spec)
# Register before executing so dataclass/typing machinery can resolve __module__.
sys.modules["pretrain_model"] = _module
_spec.loader.exec_module(_module)

GPT = _module.GPT
GPTConfig = _module.GPTConfig
Block = _module.Block
CausalSelfAttention = _module.CausalSelfAttention
RMSNorm = _module.RMSNorm
SwiGLU = _module.SwiGLU
apply_rope = _module.apply_rope
alibi_slopes = _module.alibi_slopes

__all__ = [
    "GPT",
    "GPTConfig",
    "Block",
    "CausalSelfAttention",
    "RMSNorm",
    "SwiGLU",
    "apply_rope",
    "alibi_slopes",
]
