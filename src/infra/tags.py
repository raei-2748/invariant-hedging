"""Tagging helpers for dataset provenance."""
from __future__ import annotations

from typing import Dict


def build_episode_tags(
    *,
    source: str,
    split: str,
    regime: str,
    seed: int,
    stress_jump: bool,
    stress_liquidity: bool,
) -> Dict[str, str | int | bool]:
    """Return a canonical set of tags for synthetic simulation artefacts."""

    return {
        "source": str(source),
        "split": str(split),
        "regime_name": str(regime),
        "seed": int(seed),
        "stress_jump": bool(stress_jump),
        "stress_liquidity": bool(stress_liquidity),
    }
