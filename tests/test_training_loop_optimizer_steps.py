import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.core.engine import run as run_training


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class RecordingAdam(torch.optim.Adam):
    """Optimizer that records parameter snapshots for each step."""

    def __init__(self, params, *, lr: float, weight_decay: float = 0.0):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.step_history: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []

    def step(self, closure=None):  # type: ignore[override]
        before = [
            p.detach().clone()
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad
        ]
        result = super().step(closure)
        after = [
            p.detach().clone()
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad
        ]
        self.step_history.append((before, after))
        return result


@pytest.mark.slow
def test_training_loop_executes_multiple_optimizer_steps(tmp_path, monkeypatch):
    os.environ.setdefault("WANDB_MODE", "offline")
    recorded: dict[str, RecordingAdam] = {}

    def _setup_optimizer(policy, cfg, extra_params=None):
        params = [p for p in policy.parameters() if p.requires_grad]
        if extra_params is not None:
            params.extend(param for param in extra_params if param.requires_grad)
        optimizer = RecordingAdam(
            params,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
        recorded["optimizer"] = optimizer
        return optimizer

    def _setup_scheduler(*_args, **_kwargs):
        return None

    monkeypatch.setattr("src.core.engine.setup_optimizer", _setup_optimizer)
    monkeypatch.setattr("src.core.engine.setup_scheduler", _setup_scheduler)

    overrides = [
        "train.steps=4",
        "train.batch_size=8",
        "logging.log_interval=2",
        "logging.eval_interval=2",
        f"logging.local_mirror.base_dir={tmp_path.as_posix()}",
        "data.train_episodes=32",
        "data.val_episodes=8",
        "data.test_episodes=8",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR), job_name="optimizer_steps", version_base=None):
        cfg = compose(config_name="train/smoke", overrides=overrides)

    run_training(cfg)

    optimizer = recorded["optimizer"]
    # Expect one optimizer step per training iteration.
    assert len(optimizer.step_history) >= 2
    first_before = optimizer.step_history[0][0][0]
    second_before = optimizer.step_history[1][0][0]
    # Parameter values should change between steps when gradients are applied.
    assert not torch.allclose(first_before, second_before)
