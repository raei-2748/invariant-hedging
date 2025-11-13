from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from invariant_hedging.modules.data_pipeline import FeatureEngineer, unwrap_experiment_config


def test_unwrap_experiment_config_returns_train_block() -> None:
    cfg = OmegaConf.create({"train": {"seed": 7}})
    resolved = unwrap_experiment_config(cfg)
    assert isinstance(resolved, DictConfig)
    assert resolved.seed == 7


def test_feature_engineer_exposes_base_feature_names() -> None:
    engineer = FeatureEngineer()
    assert "delta" in engineer.feature_names
    assert engineer.realized_vol_window > 0
