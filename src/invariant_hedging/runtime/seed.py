"""Deterministic seeding utilities."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "torch_generator",
    "numpy_generator",
]


_NUMPY_GENERATOR: Optional[np.random.Generator] = None
_TORCH_GENERATOR: Optional[torch.Generator] = None


def _ensure_env_var(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def seed_everything(seed: int, *, numpy_default_dtype: str | np.dtype = "float64") -> torch.Generator:
    """Seed Python, NumPy, and PyTorch for full determinism.

    The function also configures deterministic-friendly environment variables and
    backend flags.  A shared :class:`torch.Generator` suitable for deterministic
    dataloaders is returned and cached for reuse via :func:`torch_generator`.
    """

    seed = int(seed)
    _ensure_env_var("PYTHONHASHSEED", str(seed))
    _ensure_env_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _ensure_env_var("CUDNN_DETERMINISTIC", "1")
    _ensure_env_var("CUDA_LAUNCH_BLOCKING", "1")

    dtype = np.dtype(numpy_default_dtype)
    _ensure_env_var("NUMPY_DEFAULT_DTYPE", dtype.name)

    random.seed(seed)
    np.random.seed(seed)

    global _NUMPY_GENERATOR, _TORCH_GENERATOR
    _NUMPY_GENERATOR = np.random.default_rng(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except AttributeError:
            pass
    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.allow_tf32 = False
        except AttributeError:
            pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, AttributeError):
        pass

    generator = torch.Generator()
    generator.manual_seed(seed)
    _TORCH_GENERATOR = generator
    return generator


def torch_generator() -> torch.Generator:
    """Return the cached torch :class:`~torch.Generator`.

    :raises RuntimeError: if :func:`seed_everything` has not been called yet.
    """

    if _TORCH_GENERATOR is None:
        raise RuntimeError("seed_everything must be called before requesting the shared torch generator.")
    return _TORCH_GENERATOR


def numpy_generator() -> np.random.Generator:
    """Return a cached NumPy :class:`~numpy.random.Generator` seeded by :func:`seed_everything`."""

    global _NUMPY_GENERATOR
    if _NUMPY_GENERATOR is None:
        _NUMPY_GENERATOR = np.random.default_rng()
    return _NUMPY_GENERATOR
