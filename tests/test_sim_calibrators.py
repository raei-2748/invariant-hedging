from pathlib import Path

import pytest
import yaml

from src.sim import calibrators


def _write_yaml(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_heston_rejects_negative_parameters(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path,
        "heston.yaml",
        {
            "base_model": "heston",
            "kappa": -0.5,
            "theta": 0.04,
            "xi": 0.3,
            "rho": 0.0,
            "v0": 0.04,
        },
    )
    with pytest.raises(ValueError, match="non-negative"):
        calibrators.load_heston(config_path)


def test_load_sabr_rejects_invalid_correlation(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path,
        "sabr.yaml",
        {
            "base_model": "sabr",
            "alpha": 0.3,
            "rho": 1.5,
            "nu": 0.5,
        },
    )
    with pytest.raises(ValueError, match=r"Correlation rho must be in \(-1, 1\)"):
        calibrators.load_sabr(config_path)


def test_compose_sim_recipe_infers_seed_from_base_config(tmp_path: Path) -> None:
    heston_path = _write_yaml(
        tmp_path,
        "heston_valid.yaml",
        {
            "base_model": "heston",
            "kappa": 1.2,
            "theta": 0.04,
            "xi": 0.3,
            "rho": -0.4,
            "v0": 0.05,
            "seed": 314,
        },
    )
    recipe = calibrators.compose_sim_recipe(
        "heston",
        heston_path,
        jumps_cfg=None,
        liq_cfg=None,
        seed=None,
    )
    assert recipe.base_model == "heston"
    assert recipe.heston is not None and recipe.heston.seed == 314
    assert recipe.seed == 314


def test_compose_sim_recipe_forbids_merton_with_sabr(tmp_path: Path) -> None:
    sabr_path = _write_yaml(
        tmp_path,
        "sabr_valid.yaml",
        {
            "base_model": "sabr",
            "alpha": 0.3,
            "rho": 0.2,
            "nu": 0.4,
            "seed": 21,
        },
    )
    merton_path = _write_yaml(
        tmp_path,
        "merton.yaml",
        {"enabled": True, "lambda_y": 0.5, "mu_j": 0.1, "sigma_j": 0.2},
    )
    with pytest.raises(ValueError, match="Merton jumps cannot be combined with SABR"):
        calibrators.compose_sim_recipe(
            "sabr",
            sabr_path,
            jumps_cfg=merton_path,
            liq_cfg=None,
            seed=None,
        )
