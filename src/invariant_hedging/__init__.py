"""Top-level package for the invariant hedging research codebase."""

from __future__ import annotations

import os
from pathlib import Path

_OPENMP_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_THREADING_LAYER": "SEQUENTIAL",
    "KMP_AFFINITY": "disabled",
    "KMP_INIT_AT_FORK": "FALSE",
}

for _key, _value in _OPENMP_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

# Allow OmegaConf to reuse keys across Hydra packages (train/runtime, etc.).
os.environ.setdefault("OMEGACONF_ALLOW_DUPLICATE_KEYS", "true")


def get_repo_root() -> Path:
    """Return the repository root, regardless of the calling working directory."""

    return Path(__file__).resolve().parents[2]


__all__ = ["get_repo_root"]
