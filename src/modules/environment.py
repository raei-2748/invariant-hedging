"""Environment interfaces aligned with the paper's invariant hedging setup."""
from __future__ import annotations

from legacy.envs.registry import EnvironmentSpec, SyntheticRegimeRegistry, register_real_anchors
from legacy.envs.single_asset import SingleAssetHedgingEnv

__all__ = [
    "EnvironmentSpec",
    "SingleAssetHedgingEnv",
    "SyntheticRegimeRegistry",
    "register_real_anchors",
]
