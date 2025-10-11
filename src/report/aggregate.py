"""Cross-seed aggregation utilities used by the reporting pipeline."""
from __future__ import annotations

import dataclasses
import glob
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import latex, plots, provenance
try:  # Optional import, some environments do not have plotly installed.
    from . import ire3d
except Exception:  # pragma: no cover - best effort fallback during optional import
    ire3d = None  # type: ignore


@dataclasses.dataclass
class AggregateResult:
    """Container for aggregated reporting artefacts."""

    seed_frame: pd.DataFrame
    stats_frame: pd.DataFrame
    outputs_dir: Path
    config: Mapping[str, object]
    selected_runs: List[Path]
    provenance: Mapping[str, object]


@dataclasses.dataclass
class CohortConfig:
    """Subset of configuration required for aggregation."""

    seeds: int
    seed_dirs: Sequence[str]
    outputs_dir: Path
    regimes_order: Sequence[str]
    confidence_level: float
    include_gfc: bool
    metrics: Mapping[str, Sequence[str]]
    qq: Mapping[str, object]
    figures: Mapping[str, object]
    latex: Mapping[str, object]
    generate_3d: bool
    ire3d: Mapping[str, object]


def _load_report_config(raw_config: Mapping[str, object], output_override: Optional[Path] = None) -> CohortConfig:
    report_cfg = raw_config.get("report")
    if not isinstance(report_cfg, Mapping):  # pragma: no cover - guard against malformed configs
        raise ValueError("Configuration is missing the 'report' block")

    outputs_dir = Path(output_override) if output_override else Path(report_cfg.get("outputs_dir", "outputs/report_assets"))

    return CohortConfig(
        seeds=int(report_cfg.get("seeds", 30)),
        seed_dirs=tuple(str(p) for p in report_cfg.get("seed_dirs", ["runs/*"])),
        outputs_dir=outputs_dir,
        regimes_order=tuple(report_cfg.get("regimes_order", [])),
        confidence_level=float(report_cfg.get("confidence_level", 0.95)),
        include_gfc=bool(report_cfg.get("include_gfc", True)),
        metrics={k: tuple(v) for k, v in (report_cfg.get("metrics", {}) or {}).items()},
        qq=dict(report_cfg.get("qq", {})),
        figures=dict(report_cfg.get("figures", {})),
        latex=dict(report_cfg.get("latex", {})),
        generate_3d=bool(report_cfg.get("generate_3d", False)),
        ire3d=dict(report_cfg.get("ire3d", {})),
    )


def _discover_seed_runs(seed_dirs: Sequence[str]) -> List[Path]:
    candidates: List[Tuple[float, Path]] = []
    for pattern in seed_dirs:
        for path in glob.glob(pattern):
            p = Path(path)
            if not p.exists() or not p.is_dir():
                continue
            try:
                ts = p.stat().st_mtime
            except OSError:  # pragma: no cover - filesystem race guard
                ts = time.time()
            candidates.append((ts, p))
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return [p for _, p in candidates]


def _safe_read_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_seed_from_name(path: Path) -> str:
    stem = path.stem
    if "seed_" in stem:
        return stem.split("seed_")[-1]
    return stem


def _melt_wide_frame(frame: pd.DataFrame, id_vars: Sequence[str], value_name: str = "value", var_name: str = "metric") -> pd.DataFrame:
    metrics = [c for c in frame.columns if c not in id_vars]
    if not metrics:
        return pd.DataFrame(columns=[*id_vars, var_name, value_name])
    return frame.melt(id_vars=id_vars, value_name=value_name, var_name=var_name)


def _records_from_final_metrics(final_metrics: Mapping[str, object], seeds: Sequence[str]) -> List[Mapping[str, object]]:
    records: List[Mapping[str, object]] = []
    if not final_metrics:
        return records

    if all(isinstance(v, Mapping) for v in final_metrics.values()):
        for regime, metrics in final_metrics.items():
            if not isinstance(metrics, Mapping):
                continue
            for metric, value in metrics.items():
                for seed in seeds or ("unknown",):
                    records.append({"seed": seed, "regime": regime, "metric": metric, "value": value})
    else:
        for metric, value in final_metrics.items():
            for seed in seeds or ("unknown",):
                records.append({"seed": seed, "regime": "global", "metric": metric, "value": value})
    return records


def _load_seed_frame(run_dir: Path) -> pd.DataFrame:
    diag_frames: List[pd.DataFrame] = []
    for csv_path in sorted(run_dir.glob("diagnostics_seed_*.csv")):
        df = pd.read_csv(csv_path)
        if "regime" not in df.columns:
            continue
        seed = _parse_seed_from_name(csv_path)
        df = df.copy()
        df["seed"] = seed
        diag_frames.append(df)

    if diag_frames:
        diag_frame = pd.concat(diag_frames, ignore_index=True)
    else:
        diag_frame = pd.DataFrame(columns=["regime", "seed"])

    diag_long = _melt_wide_frame(diag_frame, id_vars=["seed", "regime"])

    final_metrics_path = run_dir / "final_metrics.json"
    final_metrics = _safe_read_json(final_metrics_path)
    seeds = tuple(diag_frame["seed"].unique()) if not diag_frame.empty else (run_dir.name,)
    final_records = _records_from_final_metrics(final_metrics, seeds)
    final_frame = pd.DataFrame(final_records)

    frames = [f for f in (diag_long, final_frame) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["seed", "regime", "metric", "value", "run_dir"])

    combined = pd.concat(frames, ignore_index=True)
    combined["run_dir"] = str(run_dir)
    return combined


def _concat_seed_frames(run_dirs: Sequence[Path]) -> pd.DataFrame:
    frames = [_load_seed_frame(run_dir) for run_dir in run_dirs]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No diagnostic data discovered in the selected runs")
    return pd.concat(frames, ignore_index=True)


def _compute_confidence_interval(values: Sequence[float], level: float) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return (math.nan, math.nan)
    if n == 1:
        return (values[0], values[0])

    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if n <= 1 else statistics.stdev(values)
    stderr = std / math.sqrt(n)

    alpha = 1 - level
    prob = 1 - alpha / 2

    quantile = None
    try:  # Student t if available
        from scipy import stats  # type: ignore

        quantile = stats.t.ppf(prob, df=n - 1)
    except Exception:  # pragma: no cover - fallback when scipy unavailable
        try:
            from statistics import NormalDist

            quantile = NormalDist().inv_cdf(prob)
        except Exception:  # pragma: no cover
            quantile = 1.96

    half_width = quantile * stderr
    return (mean - half_width, mean + half_width)


def _compute_stats_frame(seed_frame: pd.DataFrame, regimes_order: Sequence[str], confidence_level: float) -> pd.DataFrame:
    records: List[MutableMapping[str, object]] = []
    for (metric, regime), group in seed_frame.groupby(["metric", "regime"], sort=False):
        values = group["value"].astype(float).to_numpy()
        n = len(values)
        mean = float(np.mean(values)) if n else math.nan
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        ci_low, ci_high = _compute_confidence_interval(values, confidence_level)
        records.append(
            {
                "metric": metric,
                "regime": regime,
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_half_width": (ci_high - ci_low) / 2 if not math.isnan(ci_low) and not math.isnan(ci_high) else math.nan,
                "min": float(np.min(values)) if n else math.nan,
                "max": float(np.max(values)) if n else math.nan,
                "n": n,
            }
        )

    stats_frame = pd.DataFrame.from_records(records)
    if regimes_order:
        stats_frame["regime"] = pd.Categorical(stats_frame["regime"], categories=list(regimes_order), ordered=True)
        stats_frame = stats_frame.sort_values(["metric", "regime"]).reset_index(drop=True)
    else:
        stats_frame = stats_frame.sort_values(["metric", "regime"]).reset_index(drop=True)
    return stats_frame


def _ensure_output_dirs(base: Path) -> Dict[str, Path]:
    subdirs = {
        "tables": base / "tables",
        "figures": base / "figures",
        "interactive": base / "interactive",
        "manifests": base / "manifests",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def run_aggregation(
    config: Mapping[str, object],
    *,
    lite: bool = False,
    skip_3d: bool = False,
    output_dir: Optional[Path] = None,
) -> AggregateResult:
    """Aggregate diagnostics from multiple seed runs and emit report artefacts."""

    cohort_config = _load_report_config(config, output_override=output_dir)
    selected_runs = _discover_seed_runs(cohort_config.seed_dirs)
    if not selected_runs:
        raise ValueError("No run directories discovered. Check report.seed_dirs globs.")

    seed_limit = 5 if lite else cohort_config.seeds
    selected_runs = selected_runs[:seed_limit]

    seed_frame = _concat_seed_frames(selected_runs)
    regimes_order = cohort_config.regimes_order or tuple(sorted(seed_frame["regime"].unique()))
    stats_frame = _compute_stats_frame(seed_frame, regimes_order, cohort_config.confidence_level)

    output_dirs = _ensure_output_dirs(cohort_config.outputs_dir)

    latex.write_all_tables(
        stats_frame,
        metrics_config=cohort_config.metrics,
        regimes_order=regimes_order,
        latex_config=cohort_config.latex,
        output_dir=output_dirs["tables"],
    )

    plots.generate_all_plots(
        seed_frame,
        stats_frame,
        metrics_config=cohort_config.metrics,
        regimes_order=regimes_order,
        figure_config=cohort_config.figures,
        output_dir=output_dirs["figures"],
        lite=lite,
    )

    ire_metadata: Mapping[str, object] = {}
    if cohort_config.generate_3d and not skip_3d and not lite:
        if ire3d is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("I–R–E 3D generation requested but plotly is not available")
        ire_result = ire3d.generate_all(
            seed_frame,
            config=cohort_config.ire3d,
            regimes_order=regimes_order,
            output_dirs={
                "figures": output_dirs["figures"],
                "interactive": output_dirs["interactive"],
                "tables": output_dirs["tables"],
            },
        )
        ire_metadata = ire_result.metadata

    provenance_record = provenance.write_manifest(
        output_dirs["manifests"],
        config=config,
        runs=selected_runs,
        regimes_order=regimes_order,
        confidence_level=cohort_config.confidence_level,
        ire_metadata=ire_metadata,
    )

    return AggregateResult(
        seed_frame=seed_frame,
        stats_frame=stats_frame,
        outputs_dir=cohort_config.outputs_dir,
        config=config,
        selected_runs=selected_runs,
        provenance=provenance_record,
    )
