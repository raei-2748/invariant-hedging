"""Unit tests for :func:`src.diagnostics.metrics.worst_group`."""

import pytest
import torch

from src.diagnostics.metrics import worst_group


def test_worst_group_reward_uses_min() -> None:
    rewards = [0.5, 1.0, torch.tensor(0.25)]
    result = worst_group(rewards, mode="reward")
    assert result == pytest.approx(0.25)


def test_worst_group_loss_uses_max() -> None:
    losses = [torch.tensor(0.1), torch.tensor(0.3), 0.2]
    result = worst_group(losses, mode="loss")
    assert result == pytest.approx(0.3)


def test_worst_group_handles_mixed_metric_suite() -> None:
    metrics = {
        "sharpe": {"values": [0.4, -0.2, torch.tensor(0.1)], "mode": "reward", "expected": -0.2},
        "nll": {"values": [torch.tensor(0.9), 1.1, 0.8], "mode": "loss", "expected": 1.1},
        "sortino": {"values": [0.05, 0.03, torch.tensor(0.02)], "mode": "reward", "expected": 0.02},
    }

    for metric in metrics.values():
        result = worst_group(metric["values"], mode=metric["mode"])
        assert result == pytest.approx(metric["expected"])
