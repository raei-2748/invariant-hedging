#!/usr/bin/env python3
"""Run the smoke config twice and ensure metrics are identical."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from invariant_hedging import get_repo_root
from invariant_hedging.training.engine import run as train_run

CONFIG_DIR = get_repo_root() / "configs"
RUN_ROOT = get_repo_root() / "runs/test_smoke"


def _prepare_cfg(seed: int, label: str) -> DictConfig:
    job_name = f"smoke_{seed}_{label}"
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None, job_name=job_name):
        cfg = compose(
            config_name="experiment",
            overrides=[
                "train=smoke_override",
                f"train.seed={seed}",
                f"runtime.seed={seed}",
            ],
        )
    run_dir = RUN_ROOT / label
    artifacts_dir = run_dir / "artifacts"
    outputs_dir = run_dir / "outputs"
    cfg.logging.local_mirror.base_dir = str(artifacts_dir.resolve())
    cfg.runtime.output_dir = str(outputs_dir.resolve())
    return cfg


def _run_once(seed: int, label: str) -> dict:
    run_dir = RUN_ROOT / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg = _prepare_cfg(seed, label)
    metrics_path = Path(train_run(cfg))
    return json.loads(metrics_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123, help="Random seed for both runs")
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_a = _run_once(args.seed, label="pass_a")
    metrics_b = _run_once(args.seed, label="pass_b")
    if metrics_a.keys() != metrics_b.keys():
        raise SystemExit("Deterministic smoke run emitted mismatched metric keys.")
    for key in metrics_a:
        if not np.allclose(metrics_a[key], metrics_b[key], atol=1e-7, rtol=1e-5):
            raise SystemExit(f"Deterministic smoke run diverged for metric '{key}'.")
    print("[smoke-check] Metrics match across runs.")


if __name__ == "__main__":
    main()
