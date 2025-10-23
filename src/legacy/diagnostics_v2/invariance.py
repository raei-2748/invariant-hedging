"""Invariance diagnostics (ISI, IG)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

from .utils import TrimmedStats, to_dataframe, trimmed_mean


@dataclass(frozen=True)
class ISIConfig:
    """Configuration bundle for the invariance spectrum diagnostic."""

    tau_risk: float = 1.0
    tau_cov: float = 1.0
    epsilon: float = 1e-8
    trim_ratio: float = 0.1
    weights: Mapping[str, float] | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, float] | None) -> "ISIConfig":
        if config is None:
            return cls()
        return cls(
            tau_risk=float(config.get("tau_risk", 1.0)),
            tau_cov=float(config.get("tau_cov", 1.0)),
            epsilon=float(config.get("epsilon", 1e-8)),
            trim_ratio=float(config.get("trim_ratio", 0.1)),
            weights=config.get("weights"),
        )


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


def _component_trim(values: Dict[str, float], trim_ratio: float) -> Tuple[float, TrimmedStats]:
    stats = trimmed_mean(values.values(), trim_ratio)
    mean_value = float(stats.trimmed_values.mean()) if stats.trimmed_values.size else float("nan")
    return mean_value, stats


def compute_ISI(
    rep_stats,
    grad_stats,
    risk_by_env,
    config: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Compute the invariance spectrum index and component diagnostics.

    Parameters
    ----------
    rep_stats:
        Container convertible to a DataFrame with columns ["probe_id", "dispersion"].
    grad_stats:
        Container convertible to a DataFrame with columns
        ["probe_id", "env_i", "env_j", "cosine"].
    risk_by_env:
        Container convertible to a DataFrame with columns ["probe_id", "env", "risk"].
    config:
        Mapping configuring thresholds, trimming and weights.
    """

    cfg = ISIConfig.from_mapping(config)
    weights = cfg.weights or {"C1": 1.0 / 3.0, "C2": 1.0 / 3.0, "C3": 1.0 / 3.0}

    risk_df = to_dataframe(risk_by_env, ["probe_id", "env", "risk"])
    grad_df = to_dataframe(grad_stats, ["probe_id", "env_i", "env_j", "cosine"])
    rep_df = to_dataframe(rep_stats, ["probe_id", "dispersion"])

    per_probe: Dict[str, Dict[str, float]] = {}

    # Component C1 – Global Stability
    c1_values: Dict[str, float] = {}
    if not risk_df.empty:
        grouped = risk_df.groupby("probe_id")
        for probe_id, group in grouped:
            env_grouped = group.groupby("env")["risk"].mean()
            variance = float(env_grouped.var(ddof=0)) if env_grouped.size > 1 else 0.0
            component = 1.0 - min(1.0, variance / (cfg.tau_risk + cfg.epsilon))
            c1_values[probe_id] = component
            per_probe.setdefault(probe_id, {})["C1"] = component
    else:
        c1_values = {}

    # Component C2 – Mechanistic Stability
    c2_values: Dict[str, float] = {}
    if not grad_df.empty:
        grad_df = grad_df.copy()
        grad_df["pair"] = grad_df.apply(
            lambda row: tuple(sorted((row["env_i"], row["env_j"]))), axis=1
        )
        grouped = grad_df.groupby("probe_id")
        for probe_id, group in grouped:
            pair_scores: list[float] = []
            for _, pair_group in group.groupby("pair"):
                stats = trimmed_mean(pair_group["cosine"], cfg.trim_ratio)
                if stats.trimmed_values.size:
                    pair_scores.append(float(stats.trimmed_values.mean()))
            if pair_scores:
                component = float(np.mean((np.asarray(pair_scores) + 1.0) / 2.0))
                c2_values[probe_id] = component
                per_probe.setdefault(probe_id, {})["C2"] = component
    else:
        c2_values = {}

    # Component C3 – Structural Stability
    c3_values: Dict[str, float] = {}
    if not rep_df.empty:
        for probe_id, group in rep_df.groupby("probe_id"):
            dispersion = float(group["dispersion"].mean())
            component = 1.0 - min(1.0, dispersion / (cfg.tau_cov + cfg.epsilon))
            c3_values[probe_id] = component
            per_probe.setdefault(probe_id, {})["C3"] = component
    else:
        c3_values = {}

    # Aggregate with trimming across probes
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


def compute_IG(regime_outcomes, config: Mapping[str, object] | None = None) -> Dict[str, object]:
    """Compute the invariance gap across test environments."""

    cfg = IGConfig.from_mapping(config)
    df = to_dataframe(regime_outcomes, ["env", "value"])
    if cfg.env_filter is not None:
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
