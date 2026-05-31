"""Re-export the tokenizer helper from the pre-training package.

Distillation requires teacher and student to share the exact same tokenizer
(the KL loss aligns their distributions token-by-token), which is the same one
the base model was pretrained with. Load it from `pre-training/` so there is a
single source of truth.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PRETRAIN_UTIL = Path(__file__).resolve().parent.parent / "pre-training" / "tokenizer_util.py"
_spec = importlib.util.spec_from_file_location("pretrain_tokenizer_util", _PRETRAIN_UTIL)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load tokenizer util from {_PRETRAIN_UTIL}")
_module = importlib.util.module_from_spec(_spec)
sys.modules["pretrain_tokenizer_util"] = _module
_spec.loader.exec_module(_module)

EOT_TOKEN = _module.EOT_TOKEN
BPETokenizer = _module.BPETokenizer
load_tokenizer = _module.load_tokenizer

__all__ = ["EOT_TOKEN", "BPETokenizer", "load_tokenizer"]
