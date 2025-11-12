import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.core.engine import run as run_training
from src.modules.data.types import EpisodeBatch


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


def _make_episode(name: str, *, steps: int = 6, episodes: int = 32) -> EpisodeBatch:
    generator = torch.Generator().manual_seed(abs(hash(name)) % (2**32))
    base_path = torch.linspace(100.0, 101.0, steps + 1)
    noise = 0.01 * torch.randn((episodes, steps + 1), generator=generator)
    spot = base_path.unsqueeze(0) + noise
    option_price = torch.zeros_like(spot)
    implied_vol = torch.full_like(spot, 0.2)
    maturity = torch.linspace(0.5, 0.0, steps + 1)
    time_to_maturity = maturity.unsqueeze(0).repeat(episodes, 1)
    meta = {
        "linear_bps": 0.0,
        "quadratic": 0.0,
        "slippage_multiplier": 1.0,
        "notional": 1.0,
    }
    return EpisodeBatch(
        spot=spot,
        option_price=option_price,
        implied_vol=implied_vol,
        time_to_maturity=time_to_maturity,
        rate=0.01,
        env_name=name,
        meta=meta,
    )


class DummyDataModule:
    def __init__(self, batches: dict[str, EpisodeBatch]):
        self._batches = batches

    def prepare(self, _split: str, env_names: list[str]):
        return {name: self._batches[name] for name in env_names}


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

    env_names = ["train_env", "val_env", "test_env"]
    batches = {name: _make_episode(name) for name in env_names}
    dummy_ctx = SimpleNamespace(
        data_module=DummyDataModule(batches),
        env_order=env_names,
        name_to_index={name: idx for idx, name in enumerate(env_names)},
        env_configs={},
        cost_configs={},
    )

    monkeypatch.setattr("src.core.engine.prepare_data_module", lambda *_args, **_kwargs: dummy_ctx)
    monkeypatch.setattr("src.core.engine.setup_optimizer", _setup_optimizer)
    monkeypatch.setattr("src.core.engine.setup_scheduler", _setup_scheduler)

    cfg = OmegaConf.create(
        {
            "train": {
                "steps": 4,
                "batch_size": 8,
                "grad_clip": 1.0,
                "seed": 0,
                "pretrain_steps": 0,
                "irm_ramp_steps": 1,
                "eval_interval": 2,
                "checkpoint_topk": 1,
                "max_trade_warning_factor": 0.0,
            },
            "model": {
                "name": "erm",
                "objective": "erm",
                "hidden_width": 32,
                "hidden_depth": 2,
                "dropout": 0.0,
                "layer_norm": False,
                "max_position": 2.0,
                "use_prev_position": True,
                "representation_dim": 8,
                "adapter_hidden": 4,
            },
            "loss": {"name": "cvar", "cvar_alpha": 0.95},
            "optimizer": {"name": "adam", "lr": 5e-4, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "logging": {
                "log_interval": 2,
                "eval_interval": 2,
                "wandb": {
                    "enabled": False,
                    "project": "invariant-hedging",
                    "entity": None,
                    "offline_ok": True,
                    "dir": tmp_path.as_posix(),
                },
                "local_mirror": {
                    "enabled": True,
                    "base_dir": tmp_path.as_posix(),
                    "metrics_file": "metrics.jsonl",
                    "final_metrics_file": "final_metrics.json",
                    "config_file": "config.yaml",
                    "checkpoints_dir": "checkpoints",
                    "artifacts_dir": "artifacts",
                    "stats_csv": "stats.csv",
                },
            },
            "data": {"name": "synthetic"},
            "envs": {
                "train": ["train_env"],
                "val": ["val_env"],
                "test": ["test_env"],
            },
            "runtime": {"device": "cpu", "mixed_precision": False},
            "irm": {"enabled": False},
        }
    )

    run_training(cfg)

    optimizer = recorded["optimizer"]
    # Expect one optimizer step per training iteration.
    assert len(optimizer.step_history) >= 2
    first_before = optimizer.step_history[0][0][0]
    second_before = optimizer.step_history[1][0][0]
    # Parameter values should change between steps when gradients are applied.
    assert not torch.allclose(first_before, second_before)
