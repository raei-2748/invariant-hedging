"""Reproduce the paper's end-to-end training and evaluation pipeline."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

PHASE1_MODELS: Sequence[str] = (
    "train/erm",
    "train/erm_reg",
    "train/irm",
    "train/groupdro",
    "train/vrex",
)


def _run_process(args: Sequence[str], *, env: dict[str, str]) -> None:
    process_env = os.environ.copy()
    process_env.update(env)
    subprocess.run(args, check=True, env=process_env)


def _ensure_train_env() -> dict[str, str]:
    env = {}
    env.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_THREADING_LAYER", os.environ.get("MKL_THREADING_LAYER", "SEQUENTIAL"))
    env.setdefault("KMP_AFFINITY", os.environ.get("KMP_AFFINITY", "disabled"))
    env.setdefault("KMP_INIT_AT_FORK", os.environ.get("KMP_INIT_AT_FORK", "FALSE"))
    env.setdefault("HIRM_TORCH_NUM_THREADS", env["OMP_NUM_THREADS"])
    env.setdefault("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "0"))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"))
    env.setdefault("CUDNN_DETERMINISTIC", os.environ.get("CUDNN_DETERMINISTIC", "1"))
    env.setdefault("CUDA_LAUNCH_BLOCKING", os.environ.get("CUDA_LAUNCH_BLOCKING", "1"))
    env.setdefault("NUMPY_DEFAULT_DTYPE", os.environ.get("NUMPY_DEFAULT_DTYPE", "float64"))
    env.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
    return env


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted((p for p in root.glob("20*") if p.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return candidates[0]


def _latest_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    candidates = sorted((p for p in ckpt_dir.glob("*.pt") if p.is_file()), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return candidates[0]


def _train(config: str, overrides: Iterable[str]) -> None:
    env = _ensure_train_env()
    cmd = [sys.executable, "experiments/run_train.py", f"--config-name={config}", *overrides]
    _run_process(cmd, env=env)


def _evaluate(config: str, checkpoint: Path, overrides: Iterable[str]) -> None:
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
    cmd = [
        sys.executable,
        "experiments/run_diagnostics.py",
        f"--config-name={config}",
        f"eval.report.checkpoint_path={checkpoint}",
        *overrides,
    ]
    _run_process(cmd, env=env)


def reproduce_phase1(overrides: Iterable[str]) -> None:
    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    for config in PHASE1_MODELS:
        print(f"[reproduce] Training {config}")
        _train(config, overrides)
        latest_run = _latest_run_dir(runs_root)
        checkpoint = _latest_checkpoint(latest_run)
        print(f"[reproduce] Evaluating {config} using {checkpoint}")
        _evaluate(config, checkpoint, overrides)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="phase1", choices=["phase1", "phase2"], help="Reproduction phase to execute")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides forwarded to train/eval")
    args = parser.parse_args(argv)

    if args.phase == "phase2":
        print("Phase 2 reproduce requires manual execution; see src/invariant_hedging/legacy/experiments_notes/phase2_plan.md.")
        return 0

    reproduce_phase1(args.overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
