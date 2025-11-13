#!/usr/bin/env python3
"""Train and evaluate a multi-seed ERM baseline and emit a summary CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from invariant_hedging import get_repo_root

REPO_ROOT = get_repo_root()
DEFAULT_RUN_DIR = REPO_ROOT / "runs"
DEFAULT_EVAL_DIR = REPO_ROOT / "runs_eval"
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "_baseline_erm_base" / "ERM_base_crisis.csv"


def parse_seeds(raw: str) -> List[int]:
    parts = raw.replace(";", ",").split(",")
    seeds: List[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def snapshot(dir_path: Path) -> set[Path]:
    if not dir_path.exists():
        return set()
    return {p.resolve() for p in dir_path.iterdir() if p.is_dir()}


def newest_dir(before: set[Path], after: set[Path]) -> Path:
    new_dirs = [p for p in after if p not in before]
    if not new_dirs:
        raise RuntimeError("Expected a new run directory but none was created")
    return max(new_dirs, key=lambda p: p.stat().st_mtime)


def run_subprocess(cmd: Sequence[str], env: dict[str, str]) -> None:
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def find_checkpoint(run_dir: Path, env: dict[str, str]) -> Path:
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "scripts" / "find_latest_checkpoint.py"), str(run_dir)]
    output = subprocess.check_output(cmd, env=env)
    return Path(output.decode().strip())


def load_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "final_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    if not isinstance(metrics, dict):
        raise ValueError(f"Unexpected metrics format in {metrics_path}")
    return metrics


def _maybe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def write_summary(rows: Sequence[dict[str, object]], summary_path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written to {summary_path}")

    metric_keys = [key for key in fieldnames if key.startswith("test/")]
    if metric_keys:
        print("Averages:")
        for key in metric_keys:
            values: List[float] = []
            for row in rows:
                val = _maybe_float(row.get(key))
                if val is not None:
                    values.append(val)
            if not values:
                continue
            mean_val = sum(values) / len(values)
            print(f"  {key}: {mean_val:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-4", help="Comma- or dash-separated list of seeds (e.g. '0-4,7')")
    parser.add_argument("--config", default="train/erm", help="Hydra config name for training")
    parser.add_argument("--eval-config", default="train/erm", help="Hydra config name for evaluation")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps per seed")
    parser.add_argument("--pretrain", type=int, default=2000, help="Pre-train steps before penalties")
    parser.add_argument("--irm-ramp", type=int, default=1000, help="IRM ramp steps to keep overrides in sync")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Where to write the CSV summary")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Directory where Hydra writes train runs")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Directory where Hydra writes eval runs")
    parser.add_argument(
        "--max-trade-warning-factor",
        type=float,
        default=1.2,
        help="Multiplier applied to model.max_position for trade spike warnings",
    )
    parser.add_argument("--keep-eval", action="store_true", help="Keep eval directories (default removes them after summarising)")
    args, extra = parser.parse_known_args()

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided")

    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    run_rows: List[dict[str, object]] = []

    for seed in seeds:
        print(f"=== Training seed {seed} ===")
        before = snapshot(run_dir)
        train_cmd = [
            sys.executable,
            "experiments/run_train.py",
            f"--config-name={args.config}",
            f"train.steps={args.steps}",
            f"train.pretrain_steps={args.pretrain}",
            f"train.irm_ramp_steps={args.irm_ramp}",
            f"train.eval_interval={max(1, args.steps // 10)}",
            f"logging.eval_interval={max(1, args.steps // 10)}",
            "train.checkpoint_topk=1",
            f"train.seed={seed}",
            f"train.max_trade_warning_factor={args.max_trade_warning_factor}",
        ]
        train_cmd.extend(extra)
        run_subprocess(train_cmd, env)
        after = snapshot(run_dir)
        run_path = newest_dir(before, after)
        print(f"→ Run stored at {run_path}")

        checkpoint_path = find_checkpoint(run_path, env)
        print(f"→ Using checkpoint {checkpoint_path}")

        print(f"=== Evaluating seed {seed} ===")
        eval_run_base = eval_dir / f"seed_{seed}"
        eval_cmd = [
            sys.executable,
            "experiments/run_diagnostics.py",
            f"--config-name={args.eval_config}",
            f"eval.report.checkpoint_path={checkpoint_path}",
            f"logging.local_mirror.base_dir={eval_run_base}",
        ]
        run_subprocess(eval_cmd, env)
        print(f"→ Eval artifacts stored at {eval_run_base}")

        metrics = load_metrics(run_path)
        row = {
            "run": run_path.name,
            "seed": seed,
            "steps": args.steps,
        }
        for key, value in metrics.items():
            row[key] = value
        run_rows.append(row)

        if not args.keep_eval and eval_run_base.exists():
            subprocess.run(["rm", "-rf", str(eval_run_base)], check=False)

    write_summary(run_rows, Path(args.summary))


if __name__ == "__main__":
    main()
