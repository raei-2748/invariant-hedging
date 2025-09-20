from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from hirm_experiment.training.engine import run_training


@hydra.main(version_base=None, config_path="../../../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_training(cfg_dict)


if __name__ == "__main__":
    main()
