"""Efficiency diagnostics (ER, TR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .utils import ensure_array, to_dataframe


@dataclass(frozen=True)
class ERConfig:
    cvar_alpha: float = 0.05
    epsilon: float = 1e-8

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> "ERConfig":
        if config is None:
            return cls()
        return cls(
            cvar_alpha=float(config.get("cvar_alpha", 0.05)),
            epsilon=float(config.get("epsilon", 1e-8)),
        )


def _lower_cvar(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return float("nan")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    sorted_vals = np.sort(values)
    tail_count = max(1, int(np.ceil(alpha * sorted_vals.size)))
    tail = sorted_vals[:tail_count]
    return float(tail.mean())


def compute_ER(pnl_series, config: Mapping[str, object] | None = None) -> Dict[str, object]:
    """Compute expected return-risk efficiency."""

    cfg = ERConfig.from_mapping(config)
    df = to_dataframe(pnl_series, ["step", "pnl"])
    if df.empty:
        return {"ER": float("nan"), "supported": False}
    pnl = df["pnl"].to_numpy(dtype=np.float64)
    mean = float(pnl.mean())
    cvar = _lower_cvar(pnl, cfg.cvar_alpha)
    denom = abs(cvar) + cfg.epsilon
    er = mean / denom if denom else float("nan")
    return {
        "ER": er,
        "mean_pnl": mean,
        "cvar": cvar,
        "alpha": cfg.cvar_alpha,
        "supported": True,
    }


@dataclass(frozen=True)
class TRConfig:
    epsilon: float = 1e-8

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> "TRConfig":
        if config is None:
            return cls()
        return cls(epsilon=float(config.get("epsilon", 1e-8)))


def _positions_to_array(positions) -> np.ndarray:
    if positions is None:
        return np.zeros((0, 0), dtype=np.float64)
    if isinstance(positions, pd.DataFrame):
        data = positions.copy()
        if "step" in data.columns:
            data = data.drop(columns=["step"])
        return data.to_numpy(dtype=np.float64)
    array = ensure_array(positions)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def compute_TR(positions, config: Mapping[str, object] | None = None) -> Dict[str, object]:
    """Compute turnover ratio from an action trajectory."""

    cfg = TRConfig.from_mapping(config)
    array = _positions_to_array(positions)
    if array.size == 0:
        return {"TR": float("nan"), "supported": False}
    if array.shape[0] < 2:
        position_norm = np.linalg.norm(array, axis=1)
        mean_position = float(position_norm.mean()) if position_norm.size else 0.0
        return {
            "TR": 0.0,
            "mean_turnover": 0.0,
            "mean_position": mean_position,
            "supported": True,
        }

    deltas = np.diff(array, axis=0)
    delta_norm = np.linalg.norm(deltas, axis=1)
    position_norm = np.linalg.norm(array, axis=1)
    mean_turnover = float(delta_norm.mean())
    mean_position = float(position_norm.mean())
    tr = mean_turnover / (mean_position + cfg.epsilon)
    return {
        "TR": tr,
        "mean_turnover": mean_turnover,
        "mean_position": mean_position,
        "supported": True,
    }
