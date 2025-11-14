"""Hydra entrypoint for crisis evaluation jobs."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from invariant_hedging import get_repo_root
from invariant_hedging.evaluation.crisis import run

CONFIG_DIR = get_repo_root() / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
