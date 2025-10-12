"""Shared helpers for CLI scripts in the repository."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bootstrap_cli_environment() -> None:
    """Ensure the project root is importable and Matplotlib is headless-safe."""

    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Default to a non-interactive backend to support headless environments
    os.environ.setdefault("MPLBACKEND", "Agg")


def parse_seed_filter(value: Optional[str]) -> Optional[list[int]]:
    """Parse a comma-delimited seed filter specification."""

    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "all":
        return None

    seeds: list[int] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        seeds.append(int(stripped))
    return seeds or None


def parse_regime_filter(value: Optional[str]) -> Optional[list[str]]:
    """Parse a comma-delimited regime filter specification."""

    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "all":
        return None

    regimes = [token.strip() for token in value.split(",") if token.strip()]
    return regimes or None


def env_override(value: Optional[str], env_var: str) -> Optional[str]:
    """Return an environment override for a CLI string option if present."""

    override = os.environ.get(env_var)
    if override is not None and override.strip():
        return override
    return value

