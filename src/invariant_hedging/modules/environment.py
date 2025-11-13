"""Environment interfaces aligned with the paper's invariant hedging setup."""
from __future__ import annotations

from invariant_hedging.legacy.envs.registry import EnvironmentSpec, SyntheticRegimeRegistry, register_real_anchors
from invariant_hedging.legacy.envs.single_asset import SingleAssetHedgingEnv

__all__ = [
    "EnvironmentSpec",
    "SingleAssetHedgingEnv",
    "SyntheticRegimeRegistry",
    "register_real_anchors",
]
