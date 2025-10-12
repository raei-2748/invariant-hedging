"""Tests for the synthetic regime registry."""
from __future__ import annotations

from src.envs.registry import SyntheticRegimeRegistry


CONFIG = {
    "data": {
        "regimes": {
            "bands": [
                {"name": "train_low", "vix_max": 0.10},
                {"name": "train_med", "vix_min": 0.10, "vix_max": 0.25},
                {"name": "val_high", "vix_min": 0.25, "vix_max": 0.40},
                {"name": "test_crisis", "vix_min": 0.40},
            ]
        },
        "stress": {
            "jump": {"enabled": True, "apply_to": ["val_high", "test_crisis"]},
            "liquidity": {"enabled": True, "apply_to": ["test_crisis"]},
        },
    }
}


def test_registry_contains_expected_regimes() -> None:
    registry = SyntheticRegimeRegistry(CONFIG)
    for name in ("train_low", "train_med", "val_high", "test_crisis"):
        assert name in registry.names
        spec = registry.get(name)
        if name in {"val_high", "test_crisis"}:
            assert spec.stress_jump is True
        else:
            assert spec.stress_jump is False
        if name == "test_crisis":
            assert spec.stress_liquidity is True
        else:
            assert spec.stress_liquidity is False


def test_tags_reflect_stress_flags() -> None:
    registry = SyntheticRegimeRegistry(CONFIG)
    tags = registry.tags_for(name="test_crisis", split="test", seed=42)
    assert tags["source"] == "sim"
    assert tags["regime_name"] == "test_crisis"
    assert tags["stress_jump"] is True
    assert tags["stress_liquidity"] is True
    assert tags["seed"] == 42
