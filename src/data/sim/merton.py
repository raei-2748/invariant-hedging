"""Merton jump overlay utilities for synthetic crisis stress."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JumpSummary:
    """Statistical summary of realized Merton jumps."""

    count: int
    mean: float
    std: float

    def to_dict(self) -> Dict[str, float]:
        return {"jump_count": self.count, "jump_mean": self.mean, "jump_std": self.std}


def _stable_seed(seed: int) -> int:
    return int(seed) & 0xFFFFFFFF


def overlay_merton_jumps(
    path_df: pd.DataFrame,
    lam: float,
    mu_j: float,
    sigma_j: float,
    seed_offset: int,
) -> Tuple[pd.DataFrame, JumpSummary]:
    """Apply a compound Poisson jump process on top of Heston log returns."""

    if path_df.empty:
        return path_df.copy(), JumpSummary(0, 0.0, 0.0)
    df = path_df.copy()
    steps = len(df) - 1
    if lam <= 0.0 or steps <= 0:
        df["jump_size"] = 0.0
        return df, JumpSummary(0, 0.0, 0.0)
    rng = np.random.default_rng(_stable_seed(seed_offset))
    jump_count = rng.poisson(max(lam, 0.0))
    if jump_count <= 0:
        df["jump_size"] = 0.0
        return df, JumpSummary(0, 0.0, 0.0)

    jump_steps = rng.integers(1, steps + 1, size=jump_count)
    jump_sizes = rng.normal(loc=mu_j, scale=max(sigma_j, 1e-12), size=jump_count)
    increments = np.zeros(len(df), dtype=np.float64)
    for idx, size in zip(jump_steps, jump_sizes):
        increments[int(idx)] += float(size)

    log_returns = df["log_return"].to_numpy(copy=True)
    log_returns += increments

    log_prices = np.zeros(len(df), dtype=np.float64)
    log_prices[0] = math.log(max(float(df["spot"].iloc[0]), 1e-12))
    for step in range(1, len(df)):
        log_prices[step] = log_prices[step - 1] + log_returns[step]
    df["log_return"] = log_returns
    df["spot"] = np.exp(log_prices)
    df["jump_size"] = increments

    return df, JumpSummary(
        count=int(jump_count),
        mean=float(np.mean(jump_sizes)) if jump_sizes.size else 0.0,
        std=float(np.std(jump_sizes)) if jump_sizes.size else 0.0,
    )
