import json
from pathlib import Path

import pandas as pd
import pytest
import torch

pytest.importorskip("pyarrow")

from src.diagnostics.export import DiagnosticsRunContext, gather_and_export
from src.diagnostics.schema import CANONICAL_COLUMNS, SCHEMA_VERSION, validate_diagnostics_table


def _make_batch(risk, outcome, positions, grad, representation):
    return {
        "risk": torch.as_tensor(risk, dtype=torch.float32),
        "outcome": torch.as_tensor(outcome, dtype=torch.float32),
        "positions": torch.as_tensor(positions, dtype=torch.float32),
        "grad": torch.as_tensor(grad, dtype=torch.float32),
        "representation": torch.as_tensor(representation, dtype=torch.float32),
    }


def test_export_parquet_schema_and_manifest(tmp_path):
    run_ctx = DiagnosticsRunContext(
        output_dir=tmp_path,
        seed=7,
        git_hash="deadbeef",
        exp_id="toy",
        split_name="test",
        regime_tag="baseline",
        is_eval_split=True,
        config_hash="cfg123",
        instrument="SPY",
        metric_basis="cvar",
        units={"return": "per_step", "risk": "cvar"},
    )

    probe_cfg = {
        "batches": {
            "env_a": [
                _make_batch([0.1, 0.2], [0.05, 0.06], [[0.0, 0.1], [0.1, 0.1]], [1.0, 0.0], [[0.0, 0.0], [0.0, 0.0]])
            ],
            "env_b": [
                _make_batch([0.3, 0.4], [0.02, 0.01], [[0.0, -0.1], [0.0, -0.1]], [-1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]])
            ],
        },
    }

    isi_cfg = {
        "weights": {"C1": 0.4, "C2": 0.3, "C3": 0.3},
        "c1_max_dispersion": 1.0,
        "c3_max_distance": 2.0,
    }

    parquet_path = gather_and_export(
        run_ctx,
        model=None,
        probe_cfg=probe_cfg,
        isi_cfg=isi_cfg,
        risk_fn=lambda _m, batch: batch["risk"],
        outcome_fn=lambda _m, batch: batch["outcome"],
        position_fn=lambda _m, batch: batch["positions"],
        head_gradient_fn=lambda _m, batch: batch["grad"],
        representation_fn=lambda _m, batch: batch["representation"],
    )

    assert parquet_path.exists()

    frame = pd.read_parquet(parquet_path)
    assert tuple(frame.columns) == CANONICAL_COLUMNS
    validate_diagnostics_table(frame)
    envs = set(frame[frame["metric"] == "ISI"]["env"].tolist())
    assert {"env_a", "env_b", "__overall__"}.issubset(envs)

    manifest_path = Path(run_ctx.output_dir) / "diagnostics_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    for key in [
        "seed",
        "git_hash",
        "config_hash",
        "instrument",
        "metric_basis",
        "isi_weights",
        "units",
        "created_utc",
        "schema_version",
        "diagnostics_table",
    ]:
        assert key in manifest
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["diagnostics_table"] == parquet_path.name
