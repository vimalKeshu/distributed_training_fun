"""Re-export the Muon optimizer from the pre-training package (single source)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PRETRAIN_MUON = Path(__file__).resolve().parent.parent / "pre-training" / "muon.py"
_spec = importlib.util.spec_from_file_location("pretrain_muon", _PRETRAIN_MUON)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load Muon from {_PRETRAIN_MUON}")
_module = importlib.util.module_from_spec(_spec)
sys.modules["pretrain_muon"] = _module
_spec.loader.exec_module(_module)

Muon = _module.Muon
zeropower_via_newtonschulz5 = _module.zeropower_via_newtonschulz5

__all__ = ["Muon", "zeropower_via_newtonschulz5"]
