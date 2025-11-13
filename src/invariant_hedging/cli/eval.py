"""Hydra entrypoint for evaluation runs."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from invariant_hedging import get_repo_root
from invariant_hedging.evaluation.runner import run as run_evaluation

CONFIG_DIR = Path(get_repo_root()) / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute the crisis evaluation harness."""

    run_evaluation(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
