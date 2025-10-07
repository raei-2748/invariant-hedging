#!/usr/bin/env python3
"""Compute IG, WG, and MSI diagnostics across seeds and methods."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

LOGGER = logging.getLogger("compute_diagnostics")


@dataclass
class RunMetadata:
    method: str | None
    seed: int | None
    path: Path
    config_tag: str | None

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")

def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read Hydra config files")

def _parse_methods(value: str) -> List[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if not methods:
        raise ValueError("No methods parsed from specification")
    return methods

def _parse_range(value: str) -> List[int]:
    value = value.strip()
    if ".." in value:
        lo, hi = value.split("..", 1)
        start = int(lo)
        end = int(hi)
        if end < start:
            raise ValueError("Invalid range: upper bound smaller than lower bound")
        return list(range(start, end + 1))
    return [int(token.strip()) for token in value.split(",") if token.strip()]

def _parse_envs(value: str) -> List[str]:
    if not value:
        return []
    return [token.strip().lower() for token in value.split(",") if token.strip()]

def _candidate_run_roots(default: Path) -> List[Path]:
    roots = [default]
    extra = os.environ.get("HIRM_EXTRA_RUN_DIRS")
    if extra:
        for token in extra.split(os.pathsep):
            token = token.strip()
            if token:
                roots.append(Path(token))
    return roots

_METHOD_CANONICAL = {
    "erm": "ERM",
    "erm_reg": "ERM_reg",
    "irm": "IRM",
    "hirm_head": "HIRM_Head",
    "hirm": "HIRM",
    "groupdro": "GroupDRO",
    "group_dro": "GroupDRO",
    "vrex": "V_REx",
    "v-rex": "V_REx",
    "vrd": "V_REx",
}

def _canonical_method(name: object | None) -> Optional[str]:
    if name is None:
        return None
    key = str(name).strip().lower()
    return _METHOD_CANONICAL.get(key)

def _load_yaml(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    _require_yaml()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, Mapping):
        return data
    return {}

def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse JSON from %s", path)
        return {}
    if isinstance(data, Mapping):
        return data
    return {}

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

def _extract_method_from_config(config: Mapping[str, object]) -> Optional[str]:
    model_cfg = config.get("model") if isinstance(config, Mapping) else None
    if isinstance(model_cfg, Mapping):
        if (method := _canonical_method(model_cfg.get("name"))):
            return method
        if (method := _canonical_method(model_cfg.get("objective"))):
            return method
    algorithm_cfg = config.get("algorithm") if isinstance(config, Mapping) else None
    if isinstance(algorithm_cfg, Mapping):
        if (method := _canonical_method(algorithm_cfg.get("name"))):
            return method
    if "method" in config:
        return _canonical_method(config.get("method"))
    return None

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
                experiment_cfg = config.get("experiment")
                if isinstance(experiment_cfg, Mapping):
                    tag_values = experiment_cfg.get("tags")
                    if isinstance(tag_values, (list, tuple)):
                        tags = ",".join(str(val) for val in tag_values)
                elif "tags" in config and isinstance(config["tags"], (list, tuple)):
                    tags = ",".join(str(val) for val in config["tags"])
            yield RunMetadata(method=method, seed=seed, path=run_dir, config_tag=tags)

def _normalize_key(key: str) -> str:
    return key.replace("/", "_").replace("-", "_").lower()

_METRIC_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "es90": (
        "es90",
        "cvar90",
        "cvar_90",
        "crisis_es90",
    ),
    "es95": (
        "es95",
        "cvar",
        "cvar95",
        "cvar_95",
        "crisis_cvar",
        "crisis_es95",
    ),
    "es99": (
        "es99",
        "cvar99",
        "cvar_99",
        "crisis_es99",
    ),
    "meanpnl": (
        "mean_pnl",
        "mean",
        "crisis_mean_pnl",
    ),
    "turnover": (
        "turnover",
        "crisis_turnover",
    ),
}

def _find_metric(metrics: Mapping[str, object], split: str, metric: str) -> Optional[float]:
    if not metrics:
        return None
    split_key = split.lower()
    normalized = {_normalize_key(k): v for k, v in metrics.items()}
    aliases = list(_METRIC_ALIASES.get(metric, ()))
    candidates: List[str] = []
    for alias in aliases:
        candidates.extend([
            alias,
            f"{split_key}_{alias}",
            f"test_{split_key}_{alias}",
            f"test/{split_key}_{alias}",
            f"test/{split_key}/{alias}",
        ])
    for cand in candidates:
        key = _normalize_key(cand)
        if key in normalized:
            try:
                return float(normalized[key])
            except (TypeError, ValueError):
                continue
    return None

def _load_diagnostics_record(run_dir: Path) -> Tuple[Optional[Mapping[str, object]], float]:
    latest_record: Optional[Mapping[str, object]] = None
    latest_mtime = 0.0
    for base in (run_dir, run_dir / "artifacts"):
        jsonl_path = base / "diagnostics.jsonl"
        if not jsonl_path.exists():
            continue
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except OSError:
            continue
        for raw in reversed(lines):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(record, Mapping):
                latest_record = record
                latest_mtime = max(latest_mtime, jsonl_path.stat().st_mtime)
                break
    return latest_record, latest_mtime

def _collect_env_values(
    env_metrics: Mapping[str, Mapping[str, object]] | None,
    target_names: Iterable[str],
    expected_split: Optional[str],
    risk_key: str,
) -> List[float]:
    if not env_metrics:
        return []
    names = {name.lower() for name in target_names if name}
    values: List[float] = []
    fallback: List[float] = []
    for env_name, metrics in env_metrics.items():
        if not isinstance(metrics, Mapping):
            continue
        split = str(metrics.get("split", "")).lower()
        value = metrics.get(risk_key)
        if value is None:
            value = metrics.get(risk_key.lower())
        if value is None:
            continue
        try:
            es_val = float(value)
        except (TypeError, ValueError):
            continue
        name_key = env_name.lower()
        if names and name_key not in names:
            if expected_split and split == expected_split:
                fallback.append(es_val)
            continue
        if expected_split and split and split != expected_split:
            continue
        values.append(es_val)
    if not values and expected_split:
        return fallback
    return values

def _extract_single_env_metric(
    env_metrics: Mapping[str, Mapping[str, object]] | None,
    target_names: Iterable[str],
    expected_split: Optional[str],
    risk_key: str,
) -> Optional[float]:
    candidates = _collect_env_values(env_metrics, target_names, expected_split, risk_key)
    if candidates:
        return candidates[0]
    return None

def _gap(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return max(vals) - min(vals)

def _max_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    return max(vals) if vals else math.nan

def _min_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    return min(vals) if vals else math.nan

def _compute_wg(train_vals: Iterable[float], test_vals: Iterable[float]) -> Optional[float]:
    train_list = [float(v) for v in train_vals if v is not None and not math.isnan(v)]
    test_list = [float(v) for v in test_vals if v is not None and not math.isnan(v)]
    if not train_list or not test_list:
        return None
    return max(test_list) - max(train_list)

def _empty_row(method: str, seed: int) -> Dict[str, object]:
    return {
        "method": method,
        "seed": seed,
        "ig": math.nan,
        "wg": math.nan,
        "msi": math.nan,
        "es95_crisis": math.nan,
        "es99_crisis": math.nan,
        "meanpnl_crisis": math.nan,
        "turnover_crisis": math.nan,
        "es95_train_max": math.nan,
        "es95_train_min": math.nan,
        "es95_val_high": math.nan,
        "commit": None,
        "phase": None,
        "config_tag": None,
    }

def _ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", required=True, help="Comma-separated list of methods")
    parser.add_argument("--seeds", required=True, help="Seed specification, e.g. 0..29")
    parser.add_argument("--train_envs", required=True, help="Comma-separated training environments")
    parser.add_argument("--val_envs", required=True, help="Comma-separated validation environments")
    parser.add_argument("--test_envs", required=True, help="Comma-separated test environments")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--split", default="crisis", help="Test split name for summary metrics")
    parser.add_argument("--phase", default="phase2", help="Experiment phase label")
    parser.add_argument("--commit_hash", default="UNKNOWN", help="Commit hash for provenance")
    parser.add_argument("--config_tag", default=None, help="Optional config tag override")
    parser.add_argument("--run_roots", default=None, help="Additional run directories (os.pathsep separated)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)
    args.methods = _parse_methods(args.methods)
    args.seeds = _parse_range(args.seeds)
    args.train_envs = _parse_envs(args.train_envs)
    args.val_envs = _parse_envs(args.val_envs)
    args.test_envs = _parse_envs(args.test_envs)
    return args

def _collect_diagnostics(
    roots: Sequence[Path],
    methods: Sequence[str],
    seeds: Sequence[int],
    split: str,
    train_envs: Sequence[str],
    val_envs: Sequence[str],
    test_envs: Sequence[str],
) -> Tuple[Dict[Tuple[str, int], Dict[str, object]], Dict[str, str]]:
    method_lookup = {method.upper(): method for method in methods}
    seed_set = set(seeds)
    records: Dict[Tuple[str, int], Tuple[float, Dict[str, object]]] = {}
    config_tags: Dict[str, str] = {}
    for meta in _iter_run_metadata(roots):
        if meta.method is None or meta.seed is None:
            continue
        canonical = meta.method.upper()
        method = method_lookup.get(canonical)
        if method is None:
            continue
        if meta.seed not in seed_set:
            continue
        final_metrics = _load_json(meta.path / "final_metrics.json")
        diag_record, diag_mtime = _load_diagnostics_record(meta.path)
        metrics_mtime = (meta.path / "final_metrics.json").stat().st_mtime if (meta.path / "final_metrics.json").exists() else meta.path.stat().st_mtime
        combined_mtime = max(metrics_mtime, diag_mtime)
        env_metrics = diag_record.get("env_metrics") if isinstance(diag_record, Mapping) else None
        train_vals = _collect_env_values(env_metrics, train_envs, "train", "ES95")
        test_vals = _collect_env_values(env_metrics, test_envs, "test", "ES95")
        ig = None
        if isinstance(diag_record, Mapping):
            ig = diag_record.get("IG", {}).get("ES95") if isinstance(diag_record.get("IG"), Mapping) else None
        if ig is None:
            ig = _gap(train_vals)
        wg = None
        if isinstance(diag_record, Mapping):
            wg = diag_record.get("WG", {}).get("ES95") if
