"""Minimal driver for the head-only HIRM objective."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from train.loop import build_config, run_training


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Top-level configuration must be a mapping.")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/hirm.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    raw = _load_config(args.config)
    train_cfg = raw.setdefault("train", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    config = build_config(raw)
    run_dir = run_training(config, base_dir=args.base_dir)
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
