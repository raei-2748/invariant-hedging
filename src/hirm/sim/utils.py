"""Utility helpers for simulation tests and calibrations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def make_rng(seed: int) -> np.random.Generator:
    """Return a numpy Generator seeded for deterministic simulations."""

    return np.random.default_rng(int(seed))


def annualize_mean(log_returns: np.ndarray, year_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualize the mean of log returns."""

    return float(np.mean(log_returns) * year_days)


def annualize_variance(log_returns: np.ndarray, year_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualize the variance of log returns."""

    return float(np.var(log_returns, ddof=0) * year_days)


def ac1(series: np.ndarray) -> float:
    """Compute the lag-1 autocorrelation for a 1-D numpy array."""

    series = np.asarray(series, dtype=np.float64)
    if series.size < 2:
        return 0.0
    x = series[:-1]
    y = series[1:]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def aggregate_ac1(paths: np.ndarray) -> float:
    """Compute the average lag-1 autocorrelation across paths."""

    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2:
        raise ValueError("Expected 2-D array of shape (n_paths, steps)")
    values = [ac1(path) for path in paths]
    return float(np.mean(values))


@dataclass
class SampleMoments:
    """Container for simple sample statistics used in manifests."""

    mean: float
    variance: float
    ac1: float
    jump_rate: float
    tail_prob: float


def manifest_entry(label: str, moments: SampleMoments) -> dict:
    """Convert sample moment data to a JSON-serialisable dict."""

    return {
        "label": label,
        "annualized_mean": moments.mean,
        "annualized_variance": moments.variance,
        "variance_ac1": moments.ac1,
        "jump_rate": moments.jump_rate,
        "tail_prob_gt_3sigma": moments.tail_prob,
    }
