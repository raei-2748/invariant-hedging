from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from invariant_hedging import get_repo_root
from invariant_hedging.core.engine import run as train_run


def _run_smoke(tmp_path: Path, seed: int) -> dict:
    configs_dir = get_repo_root() / "configs"
    job_name = f"determinism_{seed}"
    with initialize_config_dir(config_dir=str(configs_dir), version_base=None, job_name=job_name):
        cfg = compose(
            config_name="experiment",
            overrides=[
                "train=smoke_override",
                f"train.seed={seed}",
                f"runtime.seed={seed}",
            ],
        )
    artifacts_root = tmp_path / f"artifacts_{seed}"
    cfg.logging.local_mirror.base_dir = str(artifacts_root)
    cfg.runtime.output_dir = str(tmp_path / f"outputs_{seed}")
    metrics_path = Path(train_run(cfg))
    return json.loads(metrics_path.read_text())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_smoke_metrics_repeatable(tmp_path: Path) -> None:
    metrics_a = _run_smoke(tmp_path, seed=123)
    metrics_b = _run_smoke(tmp_path, seed=123)
    assert metrics_a.keys() == metrics_b.keys()
    for key in metrics_a:
        assert np.allclose(
            metrics_a[key],
            metrics_b[key],
            atol=1e-7,
            rtol=1e-5,
        ), f"Metric '{key}' diverged: {metrics_a[key]} vs {metrics_b[key]}"
