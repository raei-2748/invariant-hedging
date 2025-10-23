from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

ASSET_ROOT = Path("outputs") / "report_assets"

METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "es90": (
        "es90",
        "es_90",
        "cvar90",
        "cvar_90",
        "crisis_es90",
        "es90_crisis",
        "es_90_crisis",
    ),
    "es95": (
        "es95",
        "es_95",
        "cvar",
        "cvar95",
        "cvar_95",
        "crisis_es95",
        "es95_crisis",
        "crisis_cvar",
    ),
    "es99": (
        "es99",
        "es_99",
        "cvar99",
        "cvar_99",
        "crisis_es99",
        "es99_crisis",
    ),
    "meanpnl": (
        "meanpnl",
        "mean_pnl",
        "avg_pnl",
        "mean",
        "crisis_mean_pnl",
        "pnl_mean",
    ),
    "turnover": (
        "turnover",
        "avg_turnover",
        "crisis_turnover",
    ),
}

DIAGNOSTIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "ig": ("ig", "invariance_gap", "invariance"),
    "wg": ("wg", "wasserstein", "wasserstein_gap"),
    "msi": ("msi", "stability", "market_stability"),
}


@dataclass(slots=True)
class ReportInputs:
    per_seed: pd.DataFrame
    diagnostics: pd.DataFrame
    scorecard: pd.DataFrame | None
    sources: tuple[Path, ...]


def ensure_assets_dir(base: Path | None = None) -> Path:
    """Ensure the directory used for report artefacts exists."""

    path = base or ASSET_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_header(path: Path) -> pd.Index | None:
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception as exc:  # pragma: no cover - best effort filter
        LOGGER.debug("Failed to read header from %s: %s", path, exc)
        return None
    return df.columns


def discover_per_seed_csvs(search_roots: Iterable[Path]) -> list[Path]:
    """Heuristically identify per-seed CSV files under the provided roots."""

    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        try:
            iterator = root.rglob("*.csv")
        except PermissionError:  # pragma: no cover - defensive
            LOGGER.warning("Skipping %s due to permission error", root)
            continue
        for idx, path in enumerate(iterator):
            if idx > 1000:
                LOGGER.debug("Stopping CSV discovery in %s after 1000 files", root)
                break
            if path in seen:
                continue
            seen.add(path)
            header = _read_header(path)
            if header is None:
                continue
            lowered = {col.lower() for col in header}
            if {"method", "seed"}.issubset(lowered):
                candidates.append(path)
    candidates.sort()
    return candidates


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__source__"] = str(path)
    return df


def _combine_frames(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = load_csv(path)
        except Exception as exc:  # pragma: no cover - best effort to continue
            LOGGER.warning("Skipping %s due to read error: %s", path, exc)
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, copy=False)


def _resolve_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lowered = {col.lower(): col for col in df.columns}
    for alias in aliases:
        key = alias.lower()
        if key in lowered:
            return lowered[key]
    for alias in aliases:
        token = alias.lower()
        for col_lower, col in lowered.items():
            if token in col_lower:
                return col
    return None


def resolve_metric_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        column = _resolve_column(df, aliases)
        if column:
            mapping[canonical] = column
    return mapping


def resolve_diagnostic_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in DIAGNOSTIC_ALIASES.items():
        column = _resolve_column(df, aliases)
        if column:
            mapping[canonical] = column
    return mapping


def t_critical(df: int, confidence: float = 0.95) -> float:
    if df <= 0:
        return 0.0
    alpha = 1.0 - confidence
    try:  # pragma: no cover - prefer SciPy when available
        from scipy import stats

        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    except Exception:  # pragma: no cover - graceful fallback
        from statistics import NormalDist

        return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def confidence_interval(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray([float(v) for v in values if not pd.isna(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    if math.isnan(std):
        return mean, math.nan, math.nan
    margin = t_critical(arr.size - 1) * std / math.sqrt(arr.size)
    return mean, mean - margin, mean + margin


def summarise_by_method(df: pd.DataFrame, columns: Mapping[str, str]) -> pd.DataFrame:
    if df.empty or not columns:
        return pd.DataFrame()
    if "method" not in df.columns:
        LOGGER.warning("Per-seed frame is missing 'method' column; cannot summarise")
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for method, group in df.groupby("method"):
        record: dict[str, object] = {
            "method": method,
        }
        if "seed" in group.columns:
            record["n_seeds"] = int(pd.Series(group["seed"]).dropna().nunique())
        else:
            record["n_seeds"] = int(len(group))
        for canonical, column in columns.items():
            series = pd.to_numeric(group[column], errors="coerce")
            mean, low, high = confidence_interval(series.to_numpy(dtype=float))
            record[f"{canonical}_mean"] = mean
            record[f"{canonical}_ci_low"] = low
            record[f"{canonical}_ci_high"] = high
        records.append(record)
    return pd.DataFrame.from_records(records)


def _relative_delta(value: float, baseline: float | None) -> float:
    if baseline is None or math.isnan(baseline) or baseline == 0:
        return math.nan
    if math.isnan(value):
        return math.nan
    return (value - baseline) / abs(baseline) * 100.0


def attach_deltas(scorecard: pd.DataFrame, baseline_name: str = "ERM") -> pd.DataFrame:
    if scorecard.empty:
        return scorecard
    if "method" not in scorecard.columns:
        return scorecard
    mask = scorecard["method"].str.lower() == baseline_name.lower()
    if not mask.any():
        LOGGER.warning("Baseline method '%s' not found for delta computation", baseline_name)
        return scorecard
    baseline = scorecard[mask].iloc[0]
    baseline_es95 = float(baseline.get("es95_mean", math.nan))
    baseline_meanpnl = float(baseline.get("meanpnl_mean", math.nan))
    baseline_turn = float(baseline.get("turnover_mean", math.nan))
    deltas = scorecard.copy()
    deltas["d_es95_vs_ERM_pct"] = deltas["es95_mean"].apply(lambda x: _relative_delta(x, baseline_es95))
    deltas["d_meanpnl_vs_ERM_pct"] = deltas["meanpnl_mean"].apply(lambda x: _relative_delta(x, baseline_meanpnl))
    deltas["d_turnover_vs_ERM_pct"] = deltas["turnover_mean"].apply(lambda x: _relative_delta(x, baseline_turn))
    return deltas


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git not always available
        return None


def build_scorecard(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()
    metrics = resolve_metric_columns(per_seed)
    if not metrics:
        LOGGER.warning("No metric columns found in per-seed data")
        return pd.DataFrame()
    summary = summarise_by_method(per_seed, metrics)
    if summary.empty:
        return summary
    summary = attach_deltas(summary)
    summary.insert(1, "split", "crisis")
    if "timestamp" not in summary.columns:
        summary["timestamp"] = pd.Timestamp.utcnow().isoformat()
    commit = _git_commit()
    summary["commit"] = commit
    summary["phase"] = None
    summary["config_tag"] = None
    ordered_columns = [
        "method",
        "split",
        "n_seeds",
    ]
    for name in ("es90", "es95", "es99", "meanpnl", "turnover"):
        if f"{name}_mean" in summary.columns:
            ordered_columns.extend([
                f"{name}_mean",
                f"{name}_ci_low",
                f"{name}_ci_high",
            ])
    for col in ("d_es95_vs_ERM_pct", "d_meanpnl_vs_ERM_pct", "d_turnover_vs_ERM_pct"):
        if col in summary.columns:
            ordered_columns.append(col)
    ordered_columns.extend(["commit", "phase", "config_tag", "timestamp"])
    for col in ordered_columns:
        if col not in summary.columns:
            summary[col] = math.nan if col.startswith("d_") else None
    summary = summary[ordered_columns]
    summary.sort_values("method", inplace=True, ignore_index=True)
    return summary


def extract_diagnostics(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()
    diag_cols = resolve_diagnostic_columns(per_seed)
    if not diag_cols:
        return pd.DataFrame()
    keep = ["method", "seed"]
    keep.extend(diag_cols.values())
    keep = [col for col in keep if col in per_seed.columns]
    diagnostics = per_seed[keep].copy()
    diagnostics.rename(columns={v: k for k, v in diag_cols.items()}, inplace=True)
    return diagnostics


def load_report_inputs(
    per_seed_paths: Sequence[Path] | None = None,
    scorecard_path: Path | None = None,
    search_roots: Sequence[Path] | None = None,
) -> ReportInputs:
    roots = search_roots or (Path("outputs"), Path("runs"))
    if per_seed_paths is None:
        per_seed_paths = tuple(discover_per_seed_csvs(roots))
    else:
        per_seed_paths = tuple(per_seed_paths)
    per_seed = _combine_frames(per_seed_paths)
    diagnostics = extract_diagnostics(per_seed)
    scorecard = None
    if scorecard_path is not None and scorecard_path.exists():
        try:
            scorecard = pd.read_csv(scorecard_path)
        except Exception as exc:  # pragma: no cover - degrade gracefully
            LOGGER.warning("Failed to load scorecard from %s: %s", scorecard_path, exc)
    if scorecard is None or scorecard.empty:
        scorecard = build_scorecard(per_seed)
    return ReportInputs(
        per_seed=per_seed,
        diagnostics=diagnostics,
        scorecard=scorecard,
        sources=tuple(per_seed_paths),
    )
