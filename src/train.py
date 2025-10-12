"""Compatibility wrapper exposing the Hydra entrypoint."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .pipeline.train import train_one_seed


@hydra.main(config_path="../configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    train_one_seed(cfg)


if __name__ == "__main__":
    main()
