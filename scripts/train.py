#!/usr/bin/env python3
"""Convenience wrapper around ``python -m src.train`` with sane defaults."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

_THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_THREADING_LAYER": "SEQUENTIAL",
    "KMP_AFFINITY": "disabled",
    "KMP_INIT_AT_FORK": "FALSE",
}


def _prepare_env() -> None:
    for key, value in _THREAD_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("HIRM_TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))


def _rewrite_overrides(cli_args: List[str]) -> Tuple[List[str], str, str | None]:
    overrides: List[str] = []
    config_name: str | None = None
    tag_parts: List[str] = []
    for arg in cli_args:
        if arg.startswith("config="):
            config_name = arg.split("=", 1)[1]
            tag_parts.append(config_name.replace("/", "_"))
            continue
        if arg.startswith("steps="):
            value = arg.split("=", 1)[1]
            overrides.append(f"train.steps={value}")
            tag_parts.append(f"steps{value}")
            continue
        if arg.startswith("batch_size="):
            value = arg.split("=", 1)[1]
            overrides.append(f"train.batch_size={value}")
            tag_parts.append(f"bs{value}")
            continue
        if arg.startswith("seed="):
            value = arg.split("=", 1)[1]
            overrides.append(f"train.seed={value}")
            tag_parts.append(f"seed{value}")
            continue
        overrides.append(arg)
    tag = "_".join(tag_parts) if tag_parts else "default"
    return overrides, tag, config_name


def _call_train(overrides: List[str], config_name: str | None) -> int:
    cmd = [sys.executable, "-m", "src.train"]
    if config_name:
        cmd.extend(["--config-name", config_name])
    cmd.extend(overrides)
    process = subprocess.run(cmd, check=False)
    return process.returncode


def _latest_run_symlink(tag: str) -> None:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return
    candidates = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        return
    latest = candidates[-1]
    (runs_dir / "latest").unlink(missing_ok=True)
    (runs_dir / "latest").symlink_to(latest.name)
    base_tag = tag or "default"
    sanitized = base_tag.replace("=", "_").replace("/", "_")
    tagged = runs_dir / f"latest_{sanitized}"
    if tagged.exists() or tagged.is_symlink():
        tagged.unlink()
    tagged.symlink_to(latest.name)
    counter = 0
    while True:
        numbered = runs_dir / f"latest_{counter}"
        if not numbered.exists():
            numbered.symlink_to(latest.name)
            break
        counter += 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides (e.g. train=smoke)",
    )
    parsed = parser.parse_args(argv)
    overrides, tag, config_name = _rewrite_overrides(parsed.overrides)
    _prepare_env()
    status = _call_train(overrides, config_name)
    if status == 0:
        _latest_run_symlink(tag)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
