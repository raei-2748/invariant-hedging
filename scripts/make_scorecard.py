#!/usr/bin/env python3
"""Aggregate evaluation metrics into a Phase-2 scorecard."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError as exc:  # pragma: no cover - provide helpful error later
    yaml = None  # type: ignore

LOGGER = logging.getLogger("make_scorecard")

_METHOD_CANONICAL = {
    "erm": "ERM",
    "erm_reg": "ERM_reg",
    "irm": "IRM",
    "hirm_head": "HIRM_Head",
    "hirm": "HIRM",
    "groupdro": "GroupDRO",
    "group_dro": "GroupDRO",
    "vrd": "V_REx",
    "v-rex": "V_REx",
    "vrex": "V_REx",
}

_METRIC_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "es95": (
        "es95",
        "cvar",
        "cvar95",
        "cvar_95",
        "cv_95",
        "cvar95",
        "cvar95",
        "crisis_cvar",
        "crisis_es95",
        "risk/cvar",
    ),
    "meanpnl": (
        "mean",
        "mean_pnl",
        "meanpnL",
        "avg_pnl",
        "crisis_mean_pnl",
        "pnl_mean",
    ),
    "turnover": (
        "turnover",
        "avg_turnover",
        "crisis_turnover",
    ),
}

_SCORECARD_COLUMNS: Tuple[str, ...] = (
    "method",
    "split",
    "n_seeds",
    "es95_mean",
    "es95_ci_low",
    "es95_ci_high",
    "meanpnl_mean",
    "meanpnl_ci_low",
    "meanpnl_ci_high",
    "turnover_mean",
    "turnover_ci_low",
    "turnover_ci_high",
    "d_es95_vs_ERM_pct",
    "d_meanpnl_vs_ERM_pct",
    "d_turnover_vs_ERM_pct",
    "commit",
    "phase",
    "config_tag",
    "timestamp",
)


@dataclass
class MetricRecord:
    method: str
    seed: int
    split: str
    es95: float
    meanpnl: float
    turnover: float
    source_dir: Path
    modified: float


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to run this script")


def _parse_methods(value: str) -> List[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one method must be provided")
    return methods


def _parse_seeds(value: str) -> List[int]:
    value = value.strip()
    if not value:
        raise ValueError("Seed specification must be non-empty")
    if ".." in value:
        lo, hi = value.split("..", 1)
        start = int(lo)
        end = int(hi)
        if end < start:
            raise ValueError("Invalid seed range: upper bound smaller than lower bound")
        return list(range(start, end + 1))
    seeds: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No seeds parsed from specification")
    return seeds


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Unable to parse boolean value from '{value}'")


def _t_critical(df: int, confidence: float = 0.95) -> float:
    if df <= 0:
        return 0.0
    alpha = 1.0 - confidence
    try:  # pragma: no cover - prefer SciPy when available
        from scipy import stats

        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        from statistics import NormalDist

        return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return float(statistics.fmean(values))


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0 if n == 1 else math.nan
    mu = _mean(values)
    variance = sum((x - mu) ** 2 for x in values) / (n - 1)
    return math.sqrt(max(variance, 0.0))


def _confidence_interval(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    mean_val = _mean(values)
    n = len(values)
    if n <= 1:
        return mean_val, mean_val, mean_val
    std_val = _std(values)
    if math.isnan(std_val):
        return mean_val, math.nan, math.nan
    se = std_val / math.sqrt(n)
    margin = _t_critical(n - 1, confidence) * se
    return mean_val, mean_val - margin, mean_val + margin


def _normalize_key(key: str) -> str:
    return key.replace("/", "_").replace("-", "_").lower()


def _find_metric(metrics: Mapping[str, object], split: str, metric: str) -> Optional[float]:
    split_key = split.lower()
    canonical = _METRIC_ALIASES.get(metric, ())
    candidates = []
    for alias in canonical:
        candidates.append(alias)
        candidates.append(f"{split_key}_{alias}")
        candidates.append(f"{split_key}{alias}")
        candidates.append(f"{split_key}/{alias}")
        candidates.append(f"test_{split_key}_{alias}")
        candidates.append(f"test/{split_key}_{alias}")
        candidates.append(f"test/{split_key}/{alias}")
        candidates.append(f"test_{split_key}/{alias}")
        candidates.append(f"{split_key}_{alias}_mean")
    normalized = { _normalize_key(k): v for k, v in metrics.items() }
    for cand in candidates:
        value = normalized.get(_normalize_key(cand))
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse JSON from %s", path)
        return {}


def _load_yaml(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    _require_yaml()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, Mapping):
        return data
    return {}


def _candidate_run_roots(default: Path) -> List[Path]:
    roots = [default]
    extra = os.environ.get("HIRM_EXTRA_RUN_DIRS")
    if extra:
        for token in extra.split(os.pathsep):
            token = token.strip()
            if token:
                roots.append(Path(token))
    return roots


def _canonical_method(name: str | None) -> Optional[str]:
    if not name:
        return None
    key = str(name).strip().lower()
    return _METHOD_CANONICAL.get(key)


def _extract_method_from_config(config: Mapping[str, object]) -> Optional[str]:
    model = config.get("model") if isinstance(config, Mapping) else None
    if isinstance(model, Mapping):
        if (method := _canonical_method(model.get("name"))):
            return method
        if (method := _canonical_method(model.get("objective"))):
            return method
    algo = config.get("algorithm") if isinstance(config, Mapping) else None
    if isinstance(algo, Mapping):
        if (method := _canonical_method(algo.get("name"))):
            return method
    if "method" in config:
        return _canonical_method(config.get("method"))
    return None


def _extract_seed_from_config(config: Mapping[str, object]) -> Optional[int]:
    for key in ("seed", "random_seed"):
        if key in config:
            try:
                return int(config[key])
            except (TypeError, ValueError):
                continue
    train_cfg = config.get("train") if isinstance(config, Mapping) else None
    if isinstance(train_cfg, Mapping):
        for key in ("seed", "random_seed"):
            if key in train_cfg:
                try:
                    return int(train_cfg[key])
                except (TypeError, ValueError):
                    continue
    runtime_cfg = config.get("runtime") if isinstance(config, Mapping) else None
    if isinstance(runtime_cfg, Mapping) and "seed" in runtime_cfg:
        try:
            return int(runtime_cfg["seed"])
        except (TypeError, ValueError):
            pass
    return None


@dataclass
class RunMetadata:
    method: Optional[str]
    seed: Optional[int]
    path: Path
    config_tag: Optional[str]


def _iter_run_metadata(roots: Sequence[Path]) -> Iterator[RunMetadata]:
    for root in roots:
        if not root.exists():
            continue
        for config_path in root.rglob("config.yaml"):
            run_dir = config_path.parent
            metrics_path = run_dir / "final_metrics.json"
            if not metrics_path.exists():
                continue
            config = _load_yaml(config_path)
            method = _extract_method_from_config(config)
            seed = _extract_seed_from_config(config)
            tags = None
            if isinstance(config, Mapping):
                experiment = config.get("experiment")
                if isinstance(experiment, Mapping):
                    tags_val = experiment.get("tags")
                    if isinstance(tags_val, (list, tuple)):
                        tags = ",".join(str(t) for t in tags_val)
                elif "tags" in config and isinstance(config["tags"], (list, tuple)):
                    tags = ",".join(str(t) for t in config["tags"])
            yield RunMetadata(method=method, seed=seed, path=run_dir, config_tag=tags)


def _collect_metrics(roots: Sequence[Path], methods: Sequence[str], seeds: Sequence[int], split: str) -> Tuple[List[MetricRecord], Dict[str, str]]:
    method_set = {m.upper(): m for m in methods}
    seeds_set = set(seeds)
    best: Dict[Tuple[str, int], MetricRecord] = {}
    config_tags: Dict[str, str] = {}
    for meta in _iter_run_metadata(roots):
        if meta.method is None or meta.seed is None:
            continue
        canonical_method = meta.method
        target_method = method_set.get(canonical_method.upper())
        if target_method is None:
            continue
        if meta.seed not in seeds_set:
            continue
        metrics_path = meta.path / "final_metrics.json"
        metrics = _load_json(metrics_path)
        es95 = _find_metric(metrics, split, "es95")
        meanpnl = _find_metric(metrics, split, "meanpnl")
        turnover = _find_metric(metrics, split, "turnover")
        if es95 is None or meanpnl is None or turnover is None:
            LOGGER.debug("Missing metrics for %s seed=%s at %s", canonical_method, meta.seed, meta.path)
            continue
        modified = metrics_path.stat().st_mtime if metrics_path.exists() else meta.path.stat().st_mtime
        record = MetricRecord(
            method=target_method,
            seed=int(meta.seed),
            split=split,
            es95=float(es95),
            meanpnl=float(meanpnl),
            turnover=float(turnover),
            source_dir=meta.path,
            modified=modified,
        )
        key = (target_method, int(meta.seed))
        existing = best.get(key)
        if existing is None or modified >= existing.modified:
            best[key] = record
        if meta.config_tag and target_method not in config_tags:
            config_tags[target_method] = meta.config_tag
    records = list(best.values())
    records.sort(key=lambda rec: (rec.method, rec.seed))
    return records, config_tags


def _pivot_by_method(records: Iterable[MetricRecord]) -> Dict[str, List[MetricRecord]]:
    acc: Dict[str, List[MetricRecord]] = {}
    for record in records:
        acc.setdefault(record.method, []).append(record)
    return acc


def _format_ci(mean: float, lo: float, hi: float) -> str:
    if any(math.isnan(v) for v in (mean, lo, hi)):
        return "NA"
    spread = (hi - lo) / 2.0
    return f"{mean:.4f} ± {spread:.4f}"


def _relative_delta(target: float, baseline: float) -> float:
    if baseline == 0 or math.isclose(baseline, 0.0, abs_tol=1e-12):
        return math.nan
    return (target - baseline) / baseline * 100.0


def _compute_deltas_from_records(
    grouped: Mapping[str, Sequence[MetricRecord]],
    baseline: str,
) -> Dict[str, Dict[str, float]]:
    baseline_records = grouped.get(baseline)
    if not baseline_records:
        return {}
    baseline_by_seed = {record.seed: record for record in baseline_records}

    def _stats(records: Sequence[MetricRecord], seeds: Iterable[int], attr: str) -> Optional[float]:
        values = [getattr(record, attr) for record in records if record.seed in seeds]
        return _mean(values) if values else None

    results: Dict[str, Dict[str, float]] = {}
    for method, records in grouped.items():
        shared_seeds = sorted(set(record.seed for record in records) & baseline_by_seed.keys())
        base_es95_values = [baseline_by_seed[s].es95 for s in shared_seeds if s in baseline_by_seed]
        base_meanpnl_values = [baseline_by_seed[s].meanpnl for s in shared_seeds if s in baseline_by_seed]
        base_turnover_values = [baseline_by_seed[s].turnover for s in shared_seeds if s in baseline_by_seed]
        base_es95_mean = _mean(base_es95_values) if base_es95_values else None
        base_meanpnl_mean = _mean(base_meanpnl_values) if base_meanpnl_values else None
        base_turnover_mean = _mean(base_turnover_values) if base_turnover_values else None
        es95_mean = _stats(records, shared_seeds, "es95")
        meanpnl_mean = _stats(records, shared_seeds, "meanpnl")
        turnover_mean = _stats(records, shared_seeds, "turnover")
        results[method] = {
            "d_es95_vs_ERM_pct": _relative_delta(es95_mean, base_es95_mean) if es95_mean is not None and base_es95_mean is not None else math.nan,
            "d_meanpnl_vs_ERM_pct": _relative_delta(meanpnl_mean, base_meanpnl_mean) if meanpnl_mean is not None and base_meanpnl_mean is not None else math.nan,
            "d_turnover_vs_ERM_pct": _relative_delta(turnover_mean, base_turnover_mean) if turnover_mean is not None and base_turnover_mean is not None else math.nan,
        }
    return results


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_scorecard(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        LOGGER.warning("No rows to write to %s", path)
        return
    fieldnames = list(_SCORECARD_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    headers = [
        "Method",
        "Seeds",
        "ES95",
        "Mean PnL",
        "Turnover",
        "ΔES95 vs ERM (%)",
        "ΔMeanPnL vs ERM (%)",
        "ΔTurnover vs ERM (%)",
    ]
    def format_row(row: Mapping[str, object]) -> List[str]:
        return [
            str(row.get("method", "")),
            str(row.get("n_seeds", "0")),
            _format_ci(row.get("es95_mean", math.nan), row.get("es95_ci_low", math.nan), row.get("es95_ci_high", math.nan)),
            _format_ci(row.get("meanpnl_mean", math.nan), row.get("meanpnl_ci_low", math.nan), row.get("meanpnl_ci_high", math.nan)),
            _format_ci(row.get("turnover_mean", math.nan), row.get("turnover_ci_low", math.nan), row.get("turnover_ci_high", math.nan)),
            _format_percent(row.get("d_es95_vs_ERM_pct")),
            _format_percent(row.get("d_meanpnl_vs_ERM_pct")),
            _format_percent(row.get("d_turnover_vs_ERM_pct")),
        ]

    table_rows = [headers, ["---"] * len(headers)]
    for row in rows:
        table_rows.append(format_row(row))
    with path.open("w", encoding="utf-8") as handle:
        for row in table_rows:
            handle.write("| " + " | ".join(row) + " |\n")


def _format_percent(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(val):
        return "NA"
    return f"{val:.2f}%"


def _find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    checkpoints = sorted(checkpoints_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def _launch_evaluation(method: str, seed: int, run: MetricRecord, args: argparse.Namespace) -> None:
    LOGGER.info("Would evaluate method %s seed %s using existing checkpoint at %s", method, seed, run.source_dir)
    # Placeholder for evaluation launch - actual invocation depends on experiment specifics.


def _maybe_run_evaluations(records: List[MetricRecord], args: argparse.Namespace) -> None:
    if args.read_only:
        return
    grouped = _pivot_by_method(records)
    for method, method_records in grouped.items():
        for record in method_records:
            _launch_evaluation(method, record.seed, record, args)


def _aggregate(records: Sequence[MetricRecord]) -> Tuple[List[Mapping[str, object]], Dict[str, Dict[str, object]]]:
    grouped = _pivot_by_method(records)
    stats_by_method: Dict[str, Dict[str, object]] = {}
    method_rows: List[Mapping[str, object]] = []
    for method, entries in sorted(grouped.items()):
        es95_values = [entry.es95 for entry in entries]
        meanpnl_values = [entry.meanypnl for entry in entries]
        turnover_values = [entry.turnover for entry in entries]
        es95_mean, es95_low, es95_high = _confidence_interval(es95_values)
        meanpnl_mean, meanpnl_low, meanpnl_high = _confidence_interval(meanpnl_values)
        turnover_mean, turnover_low, turnover_high = _confidence_interval(turnover_values)
        stats = {
            "method": method,
            "n_seeds": len(entries),
            "es95_mean": es95_mean,
            "es95_ci_low": es95_low,
            "es95_ci_high": es95_high,
            "meanpnl_mean": meanpnl_mean,
            "meanpnl_ci_low": meanpnl_low,
            "meanpnl_ci_high": meanpnl_high,
            "turnover_mean": turnover_mean,
            "turnover_ci_low": turnover_low,
            "turnover_ci_high": turnover_high,
        }
        stats_by_method[method] = stats
        method_rows.append(stats)
    return method_rows, stats_by_method


def _empty_stats_row(method: str) -> Dict[str, object]:
    return {
        "method": method,
        "n_seeds": 0,
        "es95_mean": math.nan,
        "es95_ci_low": math.nan,
        "es95_ci_high": math.nan,
        "meanpnl_mean": math.nan,
        "meanpnl_ci_low": math.nan,
        "meanpnl_ci_high": math.nan,
        "turnover_mean": math.nan,
        "turnover_ci_low": math.nan,
        "turnover_ci_high": math.nan,
        "d_es95_vs_ERM_pct": math.nan,
        "d_meanpnl_vs_ERM_pct": math.nan,
        "d_turnover_vs_ERM_pct": math.nan,
        "commit": None,
        "phase": None,
        "config_tag": None,
        "timestamp": None,
        "split": None,
    }


def _build_params_json(args: argparse.Namespace, config_tags: Mapping[str, str], records: Sequence[MetricRecord]) -> Dict[str, object]:
    seeds_by_method: Dict[str, List[int]] = {}
    for record in records:
        seeds_by_method.setdefault(record.method, []).append(record.seed)
    for method, seeds in seeds_by_method.items():
        seeds_by_method[method] = sorted(set(seeds))
    for method in args.methods:
        seeds_by_method.setdefault(method, [])
    params = {
        "commit": args.commit_hash,
        "phase": args.phase,
        "split": args.split,
        "seeds_requested": args.seeds,
        "methods": args.methods,
        "seed_coverage": seeds_by_method,
        "config_tags": config_tags,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_tag_override": args.config_tag,
    }
    return params


def _apply_metadata(rows: List[MutableMapping[str, object]], params: Mapping[str, object], config_tags: Mapping[str, str]) -> None:
    commit = params.get("commit")
    phase = params.get("phase")
    timestamp = params.get("timestamp")
    for row in rows:
        method = row.get("method")
        row["commit"] = commit
        row["phase"] = phase
        row["timestamp"] = timestamp
        if method in config_tags:
            row["config_tag"] = config_tags[method]
        elif params.get("config_tag_override"):
            row["config_tag"] = params["config_tag_override"]
        else:
            row.setdefault("config_tag", None)
        row["split"] = params.get("split")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", required=True, help="Comma-separated list of method identifiers")
    parser.add_argument("--seeds", required=True, help="Seed specification, e.g. 0..29")
    parser.add_argument("--split", required=True, help="Evaluation split name (e.g. crisis)")
    parser.add_argument("--outdir", required=True, help="Directory where scorecard artifacts will be stored")
    parser.add_argument("--read_only", default="true", help="Whether to skip launching evaluations (true/false)")
    parser.add_argument("--phase", required=True, help="Experiment phase label")
    parser.add_argument("--commit_hash", required=True, help="Commit hash for provenance")
    parser.add_argument("--config_tag", default=None, help="Optional config tag override for params.json")
    parser.add_argument("--run_roots", default=None, help="Optional additional run directories (os.pathsep separated)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)
    args.methods = _parse_methods(args.methods)
    args.seeds = _parse_seeds(args.seeds)
    args.read_only = _str_to_bool(str(args.read_only))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    run_roots = _candidate_run_roots(Path("runs"))
    if args.run_roots:
        for token in args.run_roots.split(os.pathsep):
            token = token.strip()
            if token:
                run_roots.append(Path(token))
    LOGGER.info("Scanning run directories: %s", ", ".join(str(p) for p in run_roots))
    records, config_tags = _collect_metrics(run_roots, args.methods, args.seeds, args.split)
    config_tags = dict(config_tags)
    if args.config_tag:
        for method in args.methods:
            config_tags.setdefault(method, args.config_tag)
    if not records:
        LOGGER.warning("No metrics found for requested configuration")
    _maybe_run_evaluations(records, args)
    grouped = _pivot_by_method(records)
    rows, _ = _aggregate(records)
    if not rows:
        LOGGER.error("Scorecard generation failed: no aggregated rows")
        return 1
    deltas = _compute_deltas_from_records(grouped, "ERM")
    rows = [dict(row) for row in rows]
    for row in rows:
        method = row.get("method")
        if method in deltas:
            row.update(deltas[method])
        else:
            row.setdefault("d_es95_vs_ERM_pct", math.nan)
            row.setdefault("d_meanpnl_vs_ERM_pct", math.nan)
            row.setdefault("d_turnover_vs_ERM_pct", math.nan)
    existing_methods = {row.get("method") for row in rows}
    for method in args.methods:
        if method not in existing_methods:
            rows.append(_empty_stats_row(method))
    params = _build_params_json(args, config_tags, records)
    _apply_metadata(rows, params, config_tags)
    outdir = Path(args.outdir)
    _ensure_outdir(outdir)
    scorecard_path = outdir / "scorecard.csv"
    _write_scorecard(scorecard_path, rows)
    markdown_path = outdir / "table_crisis.md"
    _write_markdown(markdown_path, rows)
    params_path = outdir / "params.json"
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    LOGGER.info("Scorecard written to %s", scorecard_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
