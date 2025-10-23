import json
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.core.engine import run as run_training

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _train_once(tmp_path: Path, seed: int) -> dict[str, float]:
    overrides = [
        "train.steps=5",
        "train.batch_size=8",
        f"train.seed={seed}",
        "logging.log_interval=5",
        "logging.eval_interval=5",
        f"logging.local_mirror.base_dir={tmp_path.as_posix()}",
        "data.train_episodes=64",
        "data.val_episodes=16",
        "data.test_episodes=16",
    ]
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR), job_name=f"seed_test_{seed}", version_base=None
    ):
        cfg = compose(config_name="train/smoke", overrides=overrides)
    metrics_path = run_training(cfg)
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("seed", [1234])
def test_final_metrics_match(tmp_path, seed):
    os.environ.setdefault("WANDB_MODE", "offline")
    metrics_first = _train_once(tmp_path / "runs", seed)
    metrics_second = _train_once(tmp_path / "runs", seed)
    assert metrics_first == metrics_second
