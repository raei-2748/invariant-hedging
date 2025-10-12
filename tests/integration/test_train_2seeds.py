"""Integration tests covering training determinism and diagnostics exports."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import src.eval as eval_module
from src.train import run as run_training


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

BASE_OVERRIDES = [
    "train.steps=4",
    "train.batch_size=8",
    "train.pretrain_steps=0",
    "train.irm_ramp_steps=0",
    "logging.log_interval=4",
    "logging.eval_interval=4",
    "data.train_episodes=32",
    "data.val_episodes=8",
    "data.test_episodes=8",
]


def _compose_config(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train/smoke", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    diagnostics_cfg = OmegaConf.load(CONFIG_DIR / "diagnostics" / "default.yaml")
    cfg = OmegaConf.merge(cfg, diagnostics_cfg)
    cfg.diagnostics.probe.n_batches = 1
    cfg.diagnostics.probe.batch_size = 32
    return cfg


def _train_once(tmp_path: Path, seed: int) -> tuple[dict[str, float], Path]:
    os.environ.setdefault("WANDB_MODE", "offline")
    run_base = (tmp_path / "runs").as_posix()
    overrides = BASE_OVERRIDES + [
        f"train.seed={seed}",
        f"+runtime.seed={seed}",
        f"logging.local_mirror.base_dir={run_base}",
    ]
    cfg = _compose_config(overrides)
    diagnostics_dir = tmp_path / "diagnostics"
    cfg.diagnostics.outputs.dir = diagnostics_dir.as_posix()
    metrics_path = run_training(cfg)
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return metrics, Path(metrics_path).parent


def _run_diagnostics(tmp_path: Path, checkpoint_path: Path, seed: int) -> Path:
    eval_base = (tmp_path / "eval_runs").as_posix()
    overrides = BASE_OVERRIDES + [
        f"train.seed={seed}",
        f"+runtime.seed={seed}",
        f"logging.local_mirror.base_dir={eval_base}",
        f"eval.report.checkpoint_path={checkpoint_path.as_posix()}",
    ]
    cfg = _compose_config(overrides)
    diagnostics_dir = tmp_path / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    cfg.diagnostics.outputs.dir = diagnostics_dir.as_posix()
    eval_module.main(cfg)
    return diagnostics_dir


@pytest.mark.not_heavy
def test_tiny_train_determinism_and_diagnostics(tmp_path: Path) -> None:
    metrics_seed_a, run_dir_a = _train_once(tmp_path, seed=7)
    metrics_seed_b, _ = _train_once(tmp_path, seed=7)
    metrics_seed_c, _ = _train_once(tmp_path, seed=13)

    assert metrics_seed_a.keys() == metrics_seed_b.keys() == metrics_seed_c.keys()
    for key in metrics_seed_a:
        assert metrics_seed_a[key] == pytest.approx(metrics_seed_b[key], rel=1e-6, abs=1e-6)

    differences = [abs(metrics_seed_a[key] - metrics_seed_c[key]) for key in metrics_seed_a]
    assert any(delta > 1e-5 for delta in differences)

    checkpoint_candidates = sorted((run_dir_a / "checkpoints").glob("checkpoint_*.pt"))
    assert checkpoint_candidates, "training run must emit checkpoints"
    diagnostics_root = _run_diagnostics(tmp_path, checkpoint_candidates[-1], seed=7)

    exports = [path for path in diagnostics_root.iterdir() if path.is_dir()]
    assert exports, "diagnostics export directory should not be empty"
    latest_export = max(exports, key=lambda path: path.stat().st_mtime)
    csv_files = list(latest_export.glob("diagnostics_seed_*.csv"))
    assert csv_files, "diagnostics CSV was not produced"
    csv_path = csv_files[0]
    with csv_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    assert len(lines) >= 2, "diagnostics CSV must contain header and at least one record"
    header = lines[0].split(",")
    assert {"C1_global_stability", "IG"}.issubset(header)

    manifest_path = latest_export / "diagnostics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert "created_utc" in manifest
