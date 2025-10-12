"""Deterministic seeding utilities."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .determinism import DeterminismState, seed_all

__all__ = [
    "seed_everything",
    "torch_generator",
    "numpy_generator",
    "last_state",
]


_NUMPY_GENERATOR: Optional[np.random.Generator] = None
_TORCH_GENERATOR: Optional[torch.Generator] = None
_LAST_STATE: Optional[DeterminismState] = None


def seed_everything(seed: int, *, numpy_default_dtype: str | np.dtype = "float64") -> torch.Generator:
    """Seed Python, NumPy, and PyTorch for full determinism."""

    global _NUMPY_GENERATOR, _TORCH_GENERATOR, _LAST_STATE
    state = seed_all(seed, numpy_default_dtype=numpy_default_dtype)
    _NUMPY_GENERATOR = state.numpy_generator
    _TORCH_GENERATOR = state.torch_generator
    _LAST_STATE = state
    return state.torch_generator


def torch_generator() -> torch.Generator:
    """Return the cached torch :class:`~torch.Generator`."""

    if _TORCH_GENERATOR is None:
        raise RuntimeError("seed_everything must be called before requesting the shared torch generator.")
    return _TORCH_GENERATOR


def numpy_generator() -> np.random.Generator:
    """Return a cached NumPy :class:`~numpy.random.Generator` seeded by :func:`seed_everything`."""

    global _NUMPY_GENERATOR
    if _NUMPY_GENERATOR is None:
        _NUMPY_GENERATOR = np.random.default_rng()
    return _NUMPY_GENERATOR


def last_state() -> DeterminismState:
    """Return the most recently created determinism state."""

    if _LAST_STATE is None:
        raise RuntimeError("seed_everything must be called before requesting the determinism state.")
    return _LAST_STATE
