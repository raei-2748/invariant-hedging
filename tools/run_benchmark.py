"""Benchmark harness for invariant-hedging synthetic tasks."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import json

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class BenchmarkTask:
    """Resolved benchmark task specification."""

    name: str
    method: str
    description: str
    presets: Sequence[str]
    runner: str
    params: Mapping[str, Any]


class BenchmarkError(RuntimeError):
    """Raised when the benchmark harness encounters a recoverable error."""


_RUNNERS: Dict[str, Any] = {}


def _register_runner(name: str):
    def decorator(func):
        _RUNNERS[name] = func
        return func

    return decorator


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("benchmarks/registry.yaml"),
        help="Path to the benchmark task registry.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/bench_results.parquet"),
        help="Destination parquet file for aggregated results.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="mini",
        help="Benchmark preset to execute (e.g. 'mini', 'full').",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Override ISO8601 timestamp (intended for testing).",
    )
    return parser.parse_args(argv)


def _load_registry(path: Path) -> List[BenchmarkTask]:
    if not path.exists():
        raise BenchmarkError(f"Benchmark registry not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise BenchmarkError("Benchmark registry must be a mapping")
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, Sequence):
        raise BenchmarkError("Benchmark registry missing 'tasks' sequence")

    tasks: List[BenchmarkTask] = []
    for entry in tasks_raw:
        if not isinstance(entry, Mapping):
            raise BenchmarkError("Each task entry must be a mapping")
        name = str(entry.get("name")) if "name" in entry else None
        method = str(entry.get("method", entry.get("name", "")))
        description = str(entry.get("description", ""))
        presets_raw = entry.get("presets", [])
        if isinstance(presets_raw, Sequence) and not isinstance(presets_raw, (str, bytes)):
            presets = [str(item) for item in presets_raw]
        elif presets_raw:
            presets = [str(presets_raw)]
        else:
            presets = []
        runner = str(entry.get("runner", "synthetic_linear"))
        params = entry.get("params", {})
        if name is None:
            raise BenchmarkError("Task entry missing a 'name'")
        if not isinstance(params, Mapping):
            raise BenchmarkError(f"Task '{name}' params must be a mapping")
        tasks.append(
            BenchmarkTask(
                name=name,
                method=method,
                description=description,
                presets=tuple(presets),
                runner=runner,
                params=dict(params),
            )
        )
    return tasks


def _select_tasks(tasks: Sequence[BenchmarkTask], preset: str) -> List[BenchmarkTask]:
    selected = [task for task in tasks if not task.presets or preset in task.presets]
    return selected


def _ensure_runner(name: str):
    if name not in _RUNNERS:
        raise BenchmarkError(f"Unknown runner '{name}'. Available: {sorted(_RUNNERS)}")
    return _RUNNERS[name]


def _cvar(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return float("nan")
    sorted_vals = np.sort(values)
    tail_count = max(int(np.ceil((1 - alpha) * sorted_vals.size)), 1)
    tail_slice = sorted_vals[:tail_count]
    return float(np.mean(tail_slice))


@_register_runner("synthetic_linear")
def _run_synthetic_linear(task: BenchmarkTask, preset: str) -> Dict[str, Any]:
    params = task.params
    rng = np.random.default_rng(int(params.get("dataset_seed", 0)))
    feature_dim = int(params.get("feature_dim", 8))
    n_envs = int(params.get("n_envs", 3))
    samples_per_env = int(params.get("samples_per_env", 256))
    noise = float(params.get("noise", 0.05))
    delta_scale = float(params.get("delta_scale", 0.2))
    invariance_weight = float(params.get("invariance_weight", 0.0))
    alpha = float(params.get("alpha", 0.95))

    base_weight = rng.normal(0.0, 1.0, size=feature_dim)
    env_offsets = rng.normal(0.0, delta_scale, size=(n_envs, feature_dim))

    train_sets: List[tuple[np.ndarray, np.ndarray]] = []
    eval_sets: List[tuple[np.ndarray, np.ndarray]] = []
    for env_idx in range(n_envs):
        weight = base_weight + env_offsets[env_idx]
        x_train = rng.normal(size=(samples_per_env, feature_dim))
        y_train = x_train @ weight + rng.normal(0.0, noise, size=samples_per_env)
        x_eval = rng.normal(size=(samples_per_env, feature_dim))
        y_eval = x_eval @ weight + rng.normal(0.0, noise, size=samples_per_env)
        train_sets.append((x_train, y_train))
        eval_sets.append((x_eval, y_eval))

    x_pooled = np.vstack([ds[0] for ds in train_sets])
    y_pooled = np.concatenate([ds[1] for ds in train_sets])
    pooled_weight = np.linalg.pinv(x_pooled) @ y_pooled

    env_weights = [np.linalg.pinv(x) @ y for (x, y) in train_sets]
    invariant_weight = np.mean(env_weights, axis=0)
    blend = float(np.clip(invariance_weight, 0.0, 1.0))

    method = task.method.strip().lower()
    if method.startswith("erm"):
        final_weight = pooled_weight
    elif method.startswith("hirm"):
        final_weight = (1.0 - blend) * pooled_weight + blend * invariant_weight
    else:
        final_weight = (1.0 - blend) * pooled_weight + blend * invariant_weight

    residuals: List[np.ndarray] = []
    env_rmses: List[float] = []
    env_means: List[float] = []
    for x_eval, y_eval in eval_sets:
        preds = x_eval @ final_weight
        err = y_eval - preds
        residuals.append(err)
        env_rmses.append(float(np.sqrt(np.mean(err**2))))
        env_means.append(float(np.mean(err)))

    residual_all = np.concatenate(residuals)
    pnl = -(residual_all**2)
    cvar_val = _cvar(pnl, alpha)
    var_idx = max(int(np.ceil((1 - alpha) * residual_all.size)) - 1, 0)
    var_val = float(np.sort(pnl)[var_idx])
    rmse = float(np.sqrt(np.mean(residual_all**2)))

    return {
        "preset": preset,
        "task": task.name,
        "method": task.method,
        "description": task.description,
        "runner": task.runner,
        "feature_dim": feature_dim,
        "n_envs": n_envs,
        "samples_per_env": samples_per_env,
        "noise": noise,
        "delta_scale": delta_scale,
        "invariance_weight": invariance_weight,
        "alpha": alpha,
        "cvar": cvar_val,
        "var": var_val,
        "mean_pnl": float(np.mean(pnl)),
        "rmse": rmse,
        "env_rmse_max": max(env_rmses) if env_rmses else float("nan"),
        "env_rmse_min": min(env_rmses) if env_rmses else float("nan"),
        "env_rmse_gap": (max(env_rmses) - min(env_rmses)) if env_rmses else float("nan"),
        "env_mean_error": float(np.mean(env_means)) if env_means else float("nan"),
        "weight_norm": float(np.linalg.norm(final_weight)),
        "pooled_weight_norm": float(np.linalg.norm(pooled_weight)),
        "invariant_weight_norm": float(np.linalg.norm(invariant_weight)),
    }


def _read_existing_results(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError:
        raw = path.read_text(encoding="utf-8")
        if raw.startswith("JSON::"):
            payload = json.loads(raw[len("JSON::") :])
            return pd.DataFrame(payload)
        raise
    except (ValueError, OSError):
        raw = path.read_text(encoding="utf-8")
        if raw.startswith("JSON::"):
            payload = json.loads(raw[len("JSON::") :])
            return pd.DataFrame(payload)
        raise


def _merge_results(path: Path, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = _read_existing_results(path)
        combined = existing
        if not existing.empty:
            new_keys = set(zip(new_df.get("task", []), new_df.get("preset", [])))
            if new_keys:
                existing_keys = list(zip(existing.get("task", []), existing.get("preset", [])))
                mask = [key not in new_keys for key in existing_keys]
                combined = existing.loc[mask]
        combined = pd.concat([combined, new_df], ignore_index=True)
    else:
        combined = new_df
    return combined


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        payload = json.dumps(df.to_dict(orient="records"), ensure_ascii=False)
        path.write_text("JSON::" + payload + "\n", encoding="utf-8")


def _render_leaderboard(df: pd.DataFrame, output_path: Path, generated_ts: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []
    for preset in sorted(df["preset"].unique()):
        subset = df[df["preset"] == preset].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(by="cvar", ascending=False)
        display_cols = [
            "task",
            "method",
            "cvar",
            "mean_pnl",
            "rmse",
            "samples_per_env",
            "n_envs",
            "feature_dim",
            "timestamp",
        ]
        for col in display_cols:
            if col not in subset.columns:
                subset[col] = np.nan
        table = subset[display_cols]
        table = table.rename(
            columns={
                "task": "Task",
                "method": "Method",
                "cvar": "CVaR",
                "mean_pnl": "Mean PnL",
                "rmse": "RMSE",
                "samples_per_env": "Samples/Env",
                "n_envs": "Envs",
                "feature_dim": "Features",
                "timestamp": "Timestamp",
            }
        )
        sections.append(f"<h2>Preset: {preset}</h2>\n" + table.to_html(index=False, float_format=lambda x: f"{x:.6f}"))

    html = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Invariant Hedging Benchmark Leaderboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f9fbfd; color: #1a1a1a; }}
    h1 {{ margin-bottom: 0.5rem; }}
    h2 {{ margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #cdd7e0; padding: 0.5rem; text-align: right; }}
    th {{ background: #23395d; color: #fff; }}
    td:first-child, th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Invariant Hedging Benchmark Leaderboard</h1>
  <p>Last generated: {timestamp}</p>
  {sections}
</body>
</html>
""".strip().format(timestamp=generated_ts, sections="\n".join(sections))
    output_path.write_text(html, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    tasks = _load_registry(args.registry)
    preset = args.preset
    selected = _select_tasks(tasks, preset)
    if not selected:
        available = sorted({preset for task in tasks for preset in task.presets})
        raise BenchmarkError(
            f"No tasks match preset '{preset}'. Available presets: {available}"
        )

    timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
    results: List[Dict[str, Any]] = []
    for task in selected:
        runner = _ensure_runner(task.runner)
        outcome = runner(task, preset)
        outcome["timestamp"] = timestamp
        outcome["score"] = outcome.get("cvar")
        results.append(outcome)

    combined = _merge_results(args.results, results)
    _write_parquet(args.results, combined)

    subset = pd.DataFrame(results).sort_values(by="score", ascending=False)
    display = subset[["task", "method", "cvar", "mean_pnl", "rmse", "samples_per_env"]]
    print(f"Benchmark preset '{preset}' completed at {timestamp}.")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    _render_leaderboard(combined, Path("reports/leaderboard.html"), timestamp)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
