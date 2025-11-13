"""Robustness diagnostics (WG, VR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from .utils import to_dataframe


@dataclass(frozen=True)
class WGConfig:
    alpha: float = 0.25
    epsilon: float = 1e-8

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> "WGConfig":
        if config is None:
            return cls()
        return cls(
            alpha=float(config.get("alpha", 0.25)),
            epsilon=float(config.get("epsilon", 1e-8)),
        )


def _cvar_tail(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return float("nan")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    sorted_vals = np.sort(values)
    # Evaluate the Rockafellar-Uryasev objective on candidate taus from the sample
    best = float("inf")
    best_tau = float(sorted_vals[0])
    for tau in np.concatenate(([sorted_vals[0]], sorted_vals)):
        residual = np.maximum(sorted_vals - tau, 0.0)
        value = float(tau + (1.0 / alpha) * residual.mean())
        if value < best:
            best = value
            best_tau = float(tau)
    return best, best_tau


def compute_WG(risk_stats, config: Mapping[str, object] | None = None) -> Dict[str, object]:
    """Compute worst-case generalisation via CVaR-style tail risk."""

    cfg = WGConfig.from_mapping(config)
    df = to_dataframe(risk_stats, ["env", "risk"])
    if df.empty:
        return {"WG": float("nan"), "supported": False}
    env_risks = df.groupby("env")["risk"].mean().to_numpy(dtype=np.float64)
    wg_value, tau = _cvar_tail(env_risks, cfg.alpha)
    return {
        "WG": wg_value,
        "tau": tau,
        "alpha": cfg.alpha,
        "supported": True,
    }


@dataclass(frozen=True)
class VRConfig:
    epsilon: float = 1e-8

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> "VRConfig":
        if config is None:
            return cls()
        return cls(epsilon=float(config.get("epsilon", 1e-8)))


def compute_VR(risk_series, config: Mapping[str, object] | None = None) -> Dict[str, object]:
    """Compute the volatility ratio for a rolling risk series."""

    cfg = VRConfig.from_mapping(config)
    df = to_dataframe(risk_series, ["step", "risk"])
    if df.empty:
        return {"VR": float("nan"), "supported": False}
    values = df["risk"].to_numpy(dtype=np.float64)
    std = float(values.std(ddof=0))
    mean = float(values.mean())
    vr = std / (mean + cfg.epsilon)
    return {
        "VR": vr,
        "std": std,
        "mean": mean,
        "supported": True,
    }
