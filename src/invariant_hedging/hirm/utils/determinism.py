"""Deterministic seeding utilities shared across entrypoints."""
from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np
import torch


def _maybe_get_seed(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    value = getattr(obj, "seed", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_seed(cfg: Any) -> Optional[int]:
    """Resolve a seed attribute from Hydra-style configs."""

    if cfg is None:
        return None

    for attr in ("runtime", "train"):
        seed = _maybe_get_seed(getattr(cfg, attr, None))
        if seed is not None:
            return seed
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            return None
    return None


def set_seed(seed: Optional[int]) -> None:
    """Set global seeds and deterministic flags for Python, NumPy, and PyTorch."""

    if seed is None:
        return

    seed = int(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - torch <1.8
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
