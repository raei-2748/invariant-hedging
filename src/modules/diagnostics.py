"""Diagnostic helpers aligned with the paper's robustness analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from legacy.diagnostics.helpers import detach_diagnostics, safe_eval_metric
from legacy.diagnostics.isi import (
    ISINormalizationConfig,
    compute_C1_global_stability,
    compute_C2_mechanistic_stability,
    compute_C3_structural_stability,
    compute_ISI as compute_ISI_components,
)
from legacy.diagnostics.metrics import invariant_gap, mechanistic_sensitivity, worst_group


@dataclass(frozen=True)
class TrimmedStats:
    """Container holding trimmed statistics for debugging."""

    raw_values: np.ndarray
    trimmed_values: np.ndarray
    proportion_to_cut: float

    def to_dict(self) -> dict:
        return {
            "raw_values": self.raw_values.tolist(),
            "trimmed_values": self.trimmed_values.tolist(),
            "proportion_to_cut": float(self.proportion_to_cut),
        }


def _ensure_array(values) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    if isinstance(values, (list, tuple)):
        return np.asarray(values, dtype=np.float64)
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=np.float64, copy=True)
    if np.isscalar(values):
        return np.asarray([values], dtype=np.float64)
    return np.asarray(list(values), dtype=np.float64)


def _to_dataframe(data, columns: Sequence[str]) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame(columns=columns)
    if isinstance(data, pd.DataFrame):
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise KeyError(f"Missing columns {missing} in diagnostics input")
        return data.copy()
    if isinstance(data, dict):
        return pd.DataFrame(data, columns=columns)
    if isinstance(data, Iterable):
        return pd.DataFrame(list(data), columns=columns)
    raise TypeError(f"Unsupported data format: {type(data)!r}")


def _trimmed_mean(values: Sequence[float], proportion_to_cut: float) -> TrimmedStats:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return TrimmedStats(array, array, proportion_to_cut)

    proportion = float(proportion_to_cut)
    if proportion < 0 or proportion >= 0.5:
        raise ValueError("proportion_to_cut must be in [0, 0.5)")

    sorted_vals = np.sort(array)
    k = int(np.floor(proportion * sorted_vals.size))
    if k == 0:
        trimmed = sorted_vals
    else:
        trimmed = sorted_vals[k:-k] if sorted_vals.size > 2 * k else np.array([], dtype=np.float64)
    return TrimmedStats(sorted_vals, trimmed, proportion)


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


@dataclass(frozen=True)
class TRConfig:
    epsilon: float = 1e-8

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> "TRConfig":
        if config is None:
            return cls()
        return cls(epsilon=float(config.get("epsilon", 1e-8)))


def compute_ER(pnl_series, config: Mapping[str, object] | None = None) -> Dict[str, object]:
    """Compute expected return-risk efficiency."""

    cfg = ERConfig.from_mapping(config)
    df = _to_dataframe(pnl_series, ["step", "pnl"])
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


def _positions_to_array(positions) -> np.ndarray:
    if positions is None:
        return np.zeros((0, 0), dtype=np.float64)
    if isinstance(positions, pd.DataFrame):
        data = positions.copy()
        if "step" in data.columns:
            data = data.drop(columns=["step"])
        return data.to_numpy(dtype=np.float64)
    array = _ensure_array(positions)
    if array.ndim == 1:
        return array.reshape(1, -1)
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


@dataclass(frozen=True)
class ISIConfig:
    tau_risk: float = 1.0
    tau_cov: float = 1.0
    epsilon: float = 1e-8
    trim_ratio: float = 0.1
    weights: Mapping[str, float] | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> "ISIConfig":
        if config is None:
            return cls()
        return cls(
            tau_risk=float(config.get("tau_risk", 1.0)),
            tau_cov=float(config.get("tau_cov", 1.0)),
            epsilon=float(config.get("epsilon", 1e-8)),
            trim_ratio=float(config.get("trim_ratio", 0.1)),
            weights=config.get("weights"),
        )


def _component_trim(values: Dict[str, float], trim_ratio: float) -> tuple[float, TrimmedStats]:
    stats = _trimmed_mean(values.values(), trim_ratio)
    mean_value = float(stats.trimmed_values.mean()) if stats.trimmed_values.size else float("nan")
    return mean_value, stats


def compute_invariance_spectrum(
    rep_stats,
    grad_stats,
    risk_by_env,
    config: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Compute the invariance spectrum index and component diagnostics."""

    cfg = ISIConfig.from_mapping(config)
    weights = cfg.weights or {"C1": 1.0 / 3.0, "C2": 1.0 / 3.0, "C3": 1.0 / 3.0}

    risk_df = _to_dataframe(risk_by_env, ["probe_id", "env", "risk"])
    grad_df = _to_dataframe(grad_stats, ["probe_id", "env_i", "env_j", "cosine"])
    rep_df = _to_dataframe(rep_stats, ["probe_id", "dispersion"])

    per_probe: Dict[str, Dict[str, float]] = {}

    c1_values: Dict[str, float] = {}
    if not risk_df.empty:
        for probe_id, group in risk_df.groupby("probe_id"):
            env_grouped = group.groupby("env")["risk"].mean()
            variance = float(env_grouped.var(ddof=0)) if env_grouped.size > 1 else 0.0
            component = 1.0 - min(1.0, variance / (cfg.tau_risk + cfg.epsilon))
            c1_values[probe_id] = component
            per_probe.setdefault(probe_id, {})["C1"] = component

    c2_values: Dict[str, float] = {}
    if not grad_df.empty:
        grad_df = grad_df.copy()
        grad_df["pair"] = grad_df.apply(
            lambda row: tuple(sorted((row["env_i"], row["env_j"]))), axis=1
        )
        for probe_id, group in grad_df.groupby("probe_id"):
            pair_scores: list[float] = []
            for _, pair_group in group.groupby("pair"):
                stats = _trimmed_mean(pair_group["cosine"], cfg.trim_ratio)
                if stats.trimmed_values.size:
                    pair_scores.append(float((stats.trimmed_values + 1.0).mean() / 2.0))
            if pair_scores:
                component = float(np.mean(pair_scores))
                c2_values[probe_id] = component
                per_probe.setdefault(probe_id, {})["C2"] = component

    c3_values: Dict[str, float] = {}
    if not rep_df.empty:
        for probe_id, group in rep_df.groupby("probe_id"):
            dispersion = float(group["dispersion"].mean())
            component = 1.0 - min(1.0, dispersion / (cfg.tau_cov + cfg.epsilon))
            c3_values[probe_id] = component
            per_probe.setdefault(probe_id, {})["C3"] = component

    c1_mean, c1_stats = _component_trim(c1_values, cfg.trim_ratio)
    c2_mean, c2_stats = _component_trim(c2_values, cfg.trim_ratio)
    c3_mean, c3_stats = _component_trim(c3_values, cfg.trim_ratio)

    component_means = {"C1": c1_mean, "C2": c2_mean, "C3": c3_mean}
    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in weights.items():
        value = component_means.get(key, float("nan"))
        if np.isnan(value):
            continue
        weighted_sum += float(weight) * value
        weight_total += float(weight)
    ISI_value = weighted_sum / weight_total if weight_total else float("nan")

    return {
        "ISI": ISI_value,
        "weights": dict(weights),
        "trim_ratio": cfg.trim_ratio,
        "C1": c1_mean,
        "C2": c2_mean,
        "C3": c3_mean,
        "C1_stats": c1_stats.to_dict(),
        "C2_stats": c2_stats.to_dict(),
        "C3_stats": c3_stats.to_dict(),
        "per_probe": per_probe,
        "supported": bool(c1_values or c2_values or c3_values),
    }


@dataclass(frozen=True)
class IGConfig:
    tau_norm: float = 1.0
    epsilon: float = 1e-8
    env_filter: Iterable[str] | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> "IGConfig":
        if config is None:
            return cls()
        env_filter = config.get("env_filter")
        if env_filter is not None:
            env_filter = list(env_filter)
        return cls(
            tau_norm=float(config.get("tau_norm", 1.0)),
            epsilon=float(config.get("epsilon", 1e-8)),
            env_filter=env_filter,
        )


def compute_IG(regime_outcomes, config: Mapping[str, object] | None = None) -> Dict[str, object]:
    """Compute the invariance gap across test environments."""

    cfg = IGConfig.from_mapping(config)
    if isinstance(regime_outcomes, Mapping) and not isinstance(regime_outcomes, pd.DataFrame):
        env_means = [float(_ensure_array(values).mean()) for values in regime_outcomes.values() if _ensure_array(values).size]
        if not env_means:
            return {"IG": float("nan"), "IG_norm": float("nan"), "supported": False}
        values = np.asarray(env_means, dtype=np.float64)
        ig_value = float(values.std(ddof=0))
        ig_norm = ig_value / (cfg.tau_norm + cfg.epsilon)
        return {"IG": ig_value, "IG_norm": ig_norm, "supported": values.size >= 1}
    df = _to_dataframe(regime_outcomes, ["env", "value"])
    if cfg.env_filter is not None and not df.empty:
        df = df[df["env"].isin(cfg.env_filter)]
    if df.empty:
        return {"IG": float("nan"), "IG_norm": float("nan"), "supported": False}

    env_stats = df.groupby("env")["value"].mean()
    ig_value = float(env_stats.max() - env_stats.min())
    ig_norm = ig_value / (cfg.tau_norm + cfg.epsilon)
    return {
        "IG": ig_value,
        "IG_norm": ig_norm,
        "env_max": env_stats.idxmax(),
        "env_min": env_stats.idxmin(),
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
    if isinstance(risk_series, Mapping) and not isinstance(risk_series, pd.DataFrame):
        env_means = [float(_ensure_array(values).mean()) for values in risk_series.values() if _ensure_array(values).size]
        if not env_means:
            return {"VR": float("nan"), "supported": False}
        values = np.asarray(env_means, dtype=np.float64)
        variance = float(values.var(ddof=0))
        std = float(values.std(ddof=0))
        mean = float(values.mean())
        return {"VR": variance, "std": std, "mean": mean, "supported": values.size >= 1}
    df = _to_dataframe(risk_series, ["step", "risk"])
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


def _cvar_tail(values: np.ndarray, alpha: float) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    sorted_vals = np.sort(values)
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
    if isinstance(risk_stats, Mapping) and not isinstance(risk_stats, pd.DataFrame):
        env_means = [float(_ensure_array(values).mean()) for values in risk_stats.values() if _ensure_array(values).size]
        if not env_means:
            return {"WG": float("nan"), "supported": False}
        wg_value = float(np.max(env_means))
        return {"WG": wg_value, "supported": True}
    df = _to_dataframe(risk_stats, ["env", "risk"])
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


# Backwards-compatible alias for scalar aggregation.
compute_ISI = compute_ISI_components


__all__ = [
    "ISINormalizationConfig",
    "TrimmedStats",
    "compute_C1_global_stability",
    "compute_C2_mechanistic_stability",
    "compute_C3_structural_stability",
    "compute_ER",
    "compute_IG",
    "compute_ISI",
    "compute_invariance_spectrum",
    "compute_TR",
    "compute_VR",
    "compute_WG",
    "detach_diagnostics",
    "invariant_gap",
    "mechanistic_sensitivity",
    "safe_eval_metric",
    "worst_group",
]
