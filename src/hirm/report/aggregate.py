"""Aggregation utilities for reporting pipeline."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import yaml

# Public constants
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class SeedSelection:
    """Metadata describing a selected seed run."""

    run_dir: Path
    diagnostics_path: Path
    seed: int
    timestamp: float


@dataclass
class AggregateResult:
    """Container holding the raw and aggregated data."""

    raw: pd.DataFrame
    summary: pd.DataFrame
    regimes: List[str]
    selected_seeds: List[SeedSelection]
    config: Dict[str, Any]


def load_report_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration for the report pipeline."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "report" not in config:
        raise KeyError("Configuration missing 'report' section")
    return config


def resolve_seed_directories(patterns: Sequence[str]) -> List[Path]:
    """Return directories that contain diagnostics for seeds."""
    paths: List[Path] = []
    import glob

    for pattern in patterns:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw)
            if path.is_dir() and (path / "diagnostics_manifest.json").exists():
                paths.append(path)
    return sorted(set(paths))


def _load_metadata_timestamp(metadata_path: Path) -> float:
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        timestamp = data.get("timestamp") or data.get("created_at")
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
    except FileNotFoundError:
        return 0.0
    except json.JSONDecodeError:
        return 0.0
    return 0.0


def discover_seed_files(run_dir: Path) -> List[SeedSelection]:
    """Discover per-seed diagnostic CSV files for a run directory."""
    metadata_path = run_dir / "metadata.json"
    timestamp = _load_metadata_timestamp(metadata_path)
    selections: List[SeedSelection] = []
    for diag_path in sorted(run_dir.glob("diagnostics_seed_*.csv")):
        match = re.search(r"diagnostics_seed_(\d+)", diag_path.name)
        if not match:
            continue
        seed = int(match.group(1))
        stat_timestamp = diag_path.stat().st_mtime
        effective_ts = timestamp or stat_timestamp
        selections.append(
            SeedSelection(
                run_dir=run_dir,
                diagnostics_path=diag_path,
                seed=seed,
                timestamp=effective_ts,
            )
        )
    return selections


def select_seeds(
    run_dirs: Sequence[Path],
    max_seeds: Optional[int],
) -> List[SeedSelection]:
    """Select up to ``max_seeds`` seeds deterministically."""
    all_seeds: List[SeedSelection] = []
    for run_dir in run_dirs:
        all_seeds.extend(discover_seed_files(run_dir))
    all_seeds.sort(key=lambda s: (s.timestamp, s.seed, str(s.diagnostics_path)))
    if max_seeds is None:
        return all_seeds
    return all_seeds[: max_seeds]


def _melt_if_wide(df: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {c.lower() for c in df.columns}
    if "metric" in lower_cols and "value" in lower_cols:
        # Already tidy.
        return df.rename(columns={c: c.lower() for c in df.columns})
    if "regime" not in df.columns:
        raise ValueError("Diagnostics CSV must contain a 'regime' column")
    id_cols = [c for c in df.columns if c.lower() == "regime"]
    tidy = df.melt(id_vars=id_cols, var_name="metric", value_name="value")
    tidy.columns = [col.lower() for col in tidy.columns]
    return tidy


def _flatten_metrics_dict(mapping: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, Mapping):
                    for regime, metric_value in sub_value.items():
                        records.append({"metric": sub_key, "regime": regime, "value": metric_value})
                else:
                    records.append({"metric": key, "regime": sub_key, "value": sub_value})
        else:
            # scalar metric without regime context
            records.append({"metric": key, "regime": "global", "value": value})
    return records


def load_seed_dataframe(selection: SeedSelection) -> pd.DataFrame:
    """Load diagnostics and metrics for a seed selection."""
    diag_df = pd.read_csv(selection.diagnostics_path)
    diag_df = _melt_if_wide(diag_df)
    diag_df["metric"] = diag_df["metric"].astype(str)
    diag_df["regime"] = diag_df["regime"].astype(str)

    final_metrics_path = selection.run_dir / "final_metrics.json"
    records: List[Dict[str, Any]] = []
    if final_metrics_path.exists():
        with open(final_metrics_path, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                data = {}
        if isinstance(data, Mapping):
            if "metrics" in data and isinstance(data["metrics"], Mapping):
                records.extend(_flatten_metrics_dict(data["metrics"]))
            else:
                records.extend(_flatten_metrics_dict(data))
    if records:
        metrics_df = pd.DataFrame(records)
        metrics_df = metrics_df.dropna(subset=["value"]) if not metrics_df.empty else metrics_df
        if not metrics_df.empty:
            diag_df = pd.concat([diag_df, metrics_df], ignore_index=True)

    diag_df = diag_df.drop_duplicates(subset=["regime", "metric", "value"])
    diag_df["value"] = pd.to_numeric(diag_df["value"], errors="coerce")
    diag_df = diag_df.dropna(subset=["value"])
    diag_df["seed"] = selection.seed
    diag_df["run_path"] = str(selection.run_dir)

    metadata_path = selection.run_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            try:
                metadata = json.load(handle)
            except json.JSONDecodeError:
                metadata = {}
        for key, value in metadata.items():
            diag_df[f"metadata_{key}"] = value

    manifest_path = selection.run_dir / "diagnostics_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            try:
                manifest = json.load(handle)
            except json.JSONDecodeError:
                manifest = {}
        diag_df["manifest_regimes"] = json.dumps(manifest.get("regimes", []))
        diag_df["manifest_seed"] = manifest.get("seed")
    return diag_df


def ensure_metrics_exist(raw: pd.DataFrame, metrics: Iterable[str]) -> None:
    missing = sorted(set(metrics) - set(raw["metric"].unique()))
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
            # Normal approximation fallback
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
                    "metric": metric,
                    "regime": regime,
                    "block": blocks.get(metric, "other"),
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
    seeds_limit = int(report_cfg.get("seeds", 0) or 0) or None
    if lite:
        seeds_limit = min(seeds_limit or 5, 5)
    seed_dirs = report_cfg.get("seed_dirs", ["runs/*"])
    run_dirs = resolve_seed_directories(seed_dirs)
    selected_seeds = select_seeds(run_dirs, seeds_limit)
    if not selected_seeds:
        raise RuntimeError("No seed runs discovered for aggregation")

    frames: List[pd.DataFrame] = []
    for selection in selected_seeds:
        frames.append(load_seed_dataframe(selection))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["run_path", "seed", "regime", "metric"])

    regimes_order = report_cfg.get("regimes_order", sorted(raw["regime"].unique()))
    missing_regimes = [reg for reg in regimes_order if reg not in set(raw["regime"].unique())]
    if missing_regimes:
        raise KeyError(f"Missing regimes in data: {missing_regimes}")

    metrics_config = report_cfg.get("metrics", {})
    desired_metrics = [m for metrics in metrics_config.values() for m in metrics]
    if desired_metrics:
        ensure_metrics_exist(raw, desired_metrics)
    summary = compute_summary_statistics(
        raw,
        regimes_order=regimes_order,
        metrics_config=metrics_config,
        confidence_level=float(report_cfg.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)),
    )
    return AggregateResult(
        raw=raw,
        summary=summary,
        regimes=list(regimes_order),
        selected_seeds=selected_seeds,
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
    "load_report_config",
    "resolve_seed_directories",
    "select_seeds",
]
