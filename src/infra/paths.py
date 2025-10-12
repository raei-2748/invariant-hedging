"""Canonical output path helpers for tagged episode artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


def canonical_run_dir(timestamp: str, experiment: str, root: str | Path = "runs") -> Path:
    """Return the canonical run directory ``runs/<timestamp>_<experiment>``."""

    safe_experiment = experiment.replace(" ", "_")
    return Path(root) / f"{timestamp}_{safe_experiment}"


def episode_directory(run_dir: Path | str, *, seed: int, split: str, regime_name: str) -> Path:
    """Compute ``runs/.../seeds/<seed>/<split>/<regime_name>``."""

    run_path = Path(run_dir)
    return run_path / "seeds" / f"{seed}" / split / regime_name


def ensure_episode_directory(
    run_dir: Path | str, *, seed: int, split: str, regime_name: str
) -> Path:
    """Ensure the canonical episode directory exists and return it."""

    path = episode_directory(run_dir, seed=seed, split=split, regime_name=regime_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def episode_file_path(
    run_dir: Path | str,
    tags: Mapping[str, object],
    filename: str,
) -> Path:
    """Resolve a filename inside the canonical directory for the given tags."""

    split = str(tags.get("split"))
    regime_name = str(tags.get("regime_name"))
    seed = int(tags.get("seed", 0))
    base = ensure_episode_directory(run_dir, seed=seed, split=split, regime_name=regime_name)
    return base / filename
