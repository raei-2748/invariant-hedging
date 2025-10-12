"""Deterministic seeding and provenance helpers."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

__all__ = ["DeterminismState", "seed_all", "environment_variables"]


@dataclass(frozen=True)
class DeterminismState:
    """Container describing RNG state used for a deterministic run."""

    seed: int
    numpy_generator: np.random.Generator
    torch_generator: torch.Generator

    def to_dict(self) -> Dict[str, int]:
        """Return a serialisable summary of the RNG configuration."""

        return {
            "seed": self.seed,
            "numpy_seed": self.seed,
            "torch_seed": int(self.torch_generator.initial_seed()),
        }


def _ensure_env_var(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def environment_variables(seed: int) -> Dict[str, str]:
    return {
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDNN_DETERMINISTIC": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
        "NUMPY_DEFAULT_DTYPE": "float64",
    }


def _set_backend_flags() -> None:
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


def seed_all(seed: int, *, numpy_default_dtype: str | np.dtype = "float64") -> DeterminismState:
    """Seed Python, NumPy and PyTorch for determinism."""

    seed = int(seed)
    dtype = np.dtype(numpy_default_dtype)
    env_vars = environment_variables(seed)
    env_vars["NUMPY_DEFAULT_DTYPE"] = dtype.name
    for key, value in env_vars.items():
        _ensure_env_var(key, value)

    random.seed(seed)
    np.random.seed(seed)
    numpy_generator = np.random.default_rng(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _set_backend_flags()

    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)

    return DeterminismState(seed=seed, numpy_generator=numpy_generator, torch_generator=torch_generator)
