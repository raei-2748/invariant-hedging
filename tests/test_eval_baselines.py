"""Smoke tests for evaluation baselines."""
from __future__ import annotations

import torch

from src.evaluation import evaluate_crisis as eval_module
from src.modules.data.features import FeatureEngineer
from src.modules.data.types import EpisodeBatch
from src.modules.environment import SingleAssetHedgingEnv


def _make_episode_batch() -> EpisodeBatch:
    spot = torch.tensor([[100.0, 101.0, 102.5]], dtype=torch.float32)
    option = torch.tensor([[10.0, 10.5, 11.0]], dtype=torch.float32)
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


def test_evaluate_baselines_produces_risk_metrics() -> None:
    batch = _make_episode_batch()
    engineer = FeatureEngineer(realized_vol_window=2)
    scaler = engineer.fit([batch])

    env = SingleAssetHedgingEnv(env_index=0, batch=batch, feature_engineer=engineer)
    device = torch.device("cpu")
    es_alpha_list = [0.95]
    es_keys = eval_module._resolve_es_keys(es_alpha_list)

    records = eval_module._evaluate_baselines(
        ["delta", "delta_gamma"],
        {"test_env": env},
        device,
        es_alpha_list,
        0.95,
        es_keys,
        scaler,
        engineer.feature_names,
        max_position=5.0,
    )

    baselines = {record["baseline"] for record in records}
    assert baselines == {"delta", "delta_gamma"}
    for record in records:
        assert record["env"] == "test_env"
        for key in ("cvar", "turnover", "mean_pnl"):
            assert key in record
            assert record[key] is not None
        for risk_key in es_keys:
            assert risk_key in record
            assert record[risk_key] is not None
