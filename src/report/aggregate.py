"""Aggregation utilities for reporting pipeline."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import yaml

from src.diagnostics.schema import normalize_diagnostics_frame, validate_diagnostics_table

# Public constants
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass
class AggregateResult:
    """Container holding the raw and aggregated data."""

    raw: pd.DataFrame
    summary: pd.DataFrame
    regimes: List[str]
    diagnostics_path: Path
    seeds: List[int]
    algo: Optional[str]
    config: Dict[str, Any]


def load_report_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration for the report pipeline."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "report" not in config:
        raise KeyError("Configuration missing 'report' section")
    return config


def load_diagnostics_table(path: Path) -> pd.DataFrame:
    """Load and validate the canonical diagnostics parquet."""
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Reading diagnostics parquet requires 'pyarrow' or 'fastparquet' to be installed"
        ) from exc
    frame = normalize_diagnostics_frame(frame)
    validate_diagnostics_table(frame)
    return frame


def ensure_metrics_exist(raw: pd.DataFrame, metrics: Iterable[str]) -> None:
    missing = sorted(set(metrics) - set(raw["metric"].astype(str).unique()))
    if missing:
        raise KeyError(f"Missing metrics in diagnostics: {missing}")


def compute_summary_statistics(
    raw: pd.DataFrame,
    regimes_order: Sequence[str],
    metrics_config: Mapping[str, Sequence[str]],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Compute cross-seed statistics for every metric/regime pair."""

    def t_critical(df: int) -> float:
        if df <= 0:
            return 0.0
        alpha = 1 - confidence_level
        try:
            from scipy import stats  # type: ignore

            return float(stats.t.ppf(1 - alpha / 2, df))
        except Exception:  # pragma: no cover - fallback path
            from statistics import NormalDist

            return float(NormalDist().inv_cdf(1 - alpha / 2))

    blocks: Dict[str, str] = {}
    for block, metrics in metrics_config.items():
        for metric in metrics:
            blocks[metric] = block

    rows: List[Dict[str, Any]] = []
    for metric, group_metric in raw.groupby("metric"):
        for regime, group in group_metric.groupby("regime"):
            if regime not in regimes_order:
                continue
            values = group["value"].astype(float)
            n = len(values)
            mean = float(values.mean()) if n else float("nan")
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            dfree = max(n - 1, 1)
            t_val = t_critical(dfree)
            sem = std / math.sqrt(n) if n > 0 else 0.0
            ci_half = t_val * sem
            rows.append(
                {
                    "metric": str(metric),
                    "regime": str(regime),
                    "block": blocks.get(str(metric), "other"),
                    "mean": mean,
                    "std": std,
                    "ci_half_width": ci_half,
                    "ci_low": mean - ci_half,
                    "ci_high": mean + ci_half,
                    "min": float(values.min()) if n else float("nan"),
                    "max": float(values.max()) if n else float("nan"),
                    "n": n,
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["regime"] = pd.Categorical(summary["regime"], categories=list(regimes_order), ordered=True)
        summary = summary.sort_values(["block", "metric", "regime"]).reset_index(drop=True)
    return summary


def aggregate_runs(
    config: Mapping[str, Any],
    lite: bool = False,
) -> AggregateResult:
    report_cfg = config.get("report", {})
    table_value = report_cfg.get("diagnostics_table")
    if table_value is None or str(table_value).strip() in {"", "<required>"}:
        raise KeyError("report.diagnostics_table must be provided")
    diagnostics_path = Path(str(table_value))
    raw_table = load_diagnostics_table(diagnostics_path)

    split_filter = report_cfg.get("split")
    if split_filter is not None:
        raw_table = raw_table[raw_table["split"].astype(str) == str(split_filter)].copy()
    else:
        unique_splits = sorted(raw_table["split"].dropna().astype(str).unique())
        if len(unique_splits) > 1:
            raise ValueError(
                "Multiple splits present in diagnostics table; specify report.split to disambiguate"
            )

    algo_filter = report_cfg.get("algo")
    algo_values = raw_table["algo"].dropna().astype(str)
    selected_algo: Optional[str] = None
    if algo_filter is not None:
        raw_table = raw_table[algo_values == str(algo_filter)].copy()
        selected_algo = str(algo_filter)
    else:
        unique_algos = sorted(set(algo_values.tolist()))
        if len(unique_algos) > 1:
            raise ValueError("Multiple algos present in diagnostics table; specify report.algo")
        if unique_algos:
            selected_algo = unique_algos[0]

    if raw_table.empty:
        raise RuntimeError("No diagnostics rows remain after filtering")

    raw = raw_table.copy()
    raw["regime"] = raw["env"].astype(str)
    raw["metric"] = raw["metric"].astype(str)
    raw["seed"] = raw["seed"].astype(int)
    raw["value"] = raw["value"].astype(float)
    raw = raw.drop_duplicates(subset=["algo", "seed", "regime", "metric"]).reset_index(drop=True)

    seeds_available = sorted(int(seed) for seed in pd.unique(raw["seed"]))
    seeds_limit = report_cfg.get("seeds")
    if seeds_limit is not None:
        seeds_limit = int(seeds_limit)
    if lite:
        seeds_limit = min(seeds_limit or len(seeds_available), 5)
    if seeds_limit is not None:
        selected_seeds = seeds_available[: seeds_limit]
    else:
        selected_seeds = seeds_available
    raw = raw[raw["seed"].isin(selected_seeds)].copy()
    if raw.empty:
        raise RuntimeError("No diagnostics rows remain after applying seed filters")

    regimes_order = report_cfg.get("regimes_order")
    if regimes_order:
        regimes = [str(reg) for reg in regimes_order]
    else:
        regimes = sorted(raw["regime"].astype(str).unique())
    missing_regimes = [reg for reg in regimes if reg not in set(raw["regime"].unique())]
    if missing_regimes:
        raise KeyError(f"Missing regimes in data: {missing_regimes}")

    metrics_config = report_cfg.get("metrics", {})
    desired_metrics = [m for metrics in metrics_config.values() for m in metrics]
    if desired_metrics:
        ensure_metrics_exist(raw, desired_metrics)
    summary = compute_summary_statistics(
        raw,
        regimes_order=regimes,
        metrics_config=metrics_config,
        confidence_level=float(report_cfg.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)),
    )
    return AggregateResult(
        raw=raw,
        summary=summary,
        regimes=list(regimes),
        diagnostics_path=diagnostics_path,
        seeds=selected_seeds,
        algo=selected_algo,
        config=dict(config),
    )


def build_table_dataframe(summary: pd.DataFrame, metrics: Sequence[str], regimes: Sequence[str]) -> pd.DataFrame:
    filtered = summary[summary["metric"].isin(metrics)]
    pivot = (
        filtered.set_index(["metric", "regime"])
        [["mean", "ci_half_width", "std", "n"]]
        .unstack("regime")
    )
    pivot.columns = [f"{stat}_{regime}" for stat, regime in pivot.columns]
    pivot = pivot.reindex(metrics)
    pivot.index.name = "metric"
    return pivot.reset_index()


def hash_config_section(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AggregateResult",
    "aggregate_runs",
    "build_table_dataframe",
    "compute_summary_statistics",
    "hash_config_section",
    "load_diagnostics_table",
    "load_report_config",
]
