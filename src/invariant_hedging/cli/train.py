"""Hydra entrypoint for training runs."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from invariant_hedging import get_repo_root
from invariant_hedging.training.engine import run as run_training

CONFIG_DIR = Path(get_repo_root()) / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a policy under the invariant hedging configuration suite."""

    run_training(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    main()
