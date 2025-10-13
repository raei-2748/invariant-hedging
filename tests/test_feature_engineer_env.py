from __future__ import annotations

import pytest
import torch

from hirm.data.features import FeatureEngineer
from hirm.data.synthetic import EpisodeBatch
from hirm.envs.single_asset import SingleAssetHedgingEnv
from hirm import eval as eval_module


def _make_episode_batch() -> EpisodeBatch:
    spot = torch.tensor([[100.0, 102.0, 101.0]], dtype=torch.float32)
    option = torch.tensor([[10.0, 11.0, 10.0]], dtype=torch.float32)
    implied_vol = torch.full_like(spot, 0.2)
    time_to_maturity = torch.tensor([[0.5, 0.496, 0.492]], dtype=torch.float32)
    meta = {
        "linear_bps": 0.0,
        "quadratic": 0.0,
        "slippage_multiplier": 1.0,
        "notional": 1.0,
    }
    return EpisodeBatch(
        spot=spot,
        option_price=option,
        implied_vol=implied_vol,
        time_to_maturity=time_to_maturity,
        rate=0.01,
        env_name="unit-test",
        meta=meta,
    )


def test_feature_engineer_fit_and_transform_matches_manual_scaling():
    batch = _make_episode_batch()
    inventory = torch.tensor([[0.0, 0.5]], dtype=torch.float32)

    engineer = FeatureEngineer(realized_vol_window=2)
    scaler = engineer.fit([batch])

    transformed = engineer.transform(batch, inventory)
    base = engineer.base_features(batch)
    combined = torch.cat([base, inventory.unsqueeze(-1)], dim=-1)
    expected = (combined - scaler.mean) / torch.clamp(scaler.std, min=1e-6)

    torch.testing.assert_close(transformed, expected)


def test_feature_engineer_transform_requires_fit():
    batch = _make_episode_batch()
    inventory = torch.zeros((1, batch.steps), dtype=torch.float32)
    engineer = FeatureEngineer()

    with pytest.raises(RuntimeError):
        engineer.transform(batch, inventory)


def test_feature_groups_respects_configuration_overrides():
    feature_names = ["delta", "realized_vol", "inventory", "custom_feature"]
    grouping_cfg = {
        "invariants": ["realized_vol"],
        "spurious": ["delta"],
        "default": "spurious",
    }

    invariants, spurious = eval_module._feature_groups(feature_names, grouping_cfg)

    invariant_names = {feature_names[i] for i in invariants}
    spurious_names = {feature_names[i] for i in spurious}

    assert "realized_vol" in invariant_names
    assert "delta" in spurious_names
    assert "custom_feature" in spurious_names


class _StepPolicy:
    def __init__(self, actions: list[float]):
        self._actions = actions
        self._idx = 0

    def __call__(self, features: torch.Tensor, env_index: int, representation_scale=None):
        value = self._actions[self._idx]
        self._idx = min(self._idx + 1, len(self._actions) - 1)
        action = torch.full((features.shape[0], 1), value, device=features.device)
        return {"action": action}


def test_single_asset_env_simulation_tracks_basic_metrics():
    batch = _make_episode_batch()
    engineer = FeatureEngineer()
    engineer.fit([batch])

    env = SingleAssetHedgingEnv(env_index=0, batch=batch, feature_engineer=engineer)
    policy = _StepPolicy([1.0, 0.0])

    result = env.simulate(policy, torch.tensor([0]), torch.device("cpu"))

    torch.testing.assert_close(result.positions[0], torch.tensor([0.0, 1.0, 0.0]))
    assert pytest.approx(result.turnover.item(), rel=1e-6) == 2.0
    assert pytest.approx(result.max_trade.item(), rel=1e-6) == 1.0
    assert pytest.approx(result.costs.item(), rel=1e-6) == 0.0
    assert pytest.approx(result.option_pnl.item(), rel=1e-6) == 0.0
    assert pytest.approx(result.underlying_pnl.item(), rel=1e-6) == -1.0
    assert pytest.approx(result.pnl.item(), rel=1e-6) == -1.0
    assert result.probe is None
