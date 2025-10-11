"""Schema validation for diagnostics exports."""

import csv
import json
from pathlib import Path

import torch

from src.diagnostics.export import DiagnosticsRunContext, gather_and_export


def _make_batch(risk, outcome, positions, grad, representation):
    return {
        "risk": torch.as_tensor(risk, dtype=torch.float32),
        "outcome": torch.as_tensor(outcome, dtype=torch.float32),
        "positions": torch.as_tensor(positions, dtype=torch.float32),
        "grad": torch.as_tensor(grad, dtype=torch.float32),
        "representation": torch.as_tensor(representation, dtype=torch.float32),
    }


def test_export_csv_schema_and_manifest(tmp_path):
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

    csv_path = gather_and_export(
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

    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_columns = [
            "seed",
            "git_hash",
            "exp_id",
            "split_name",
            "regime_tag",
            "env_id",
            "is_eval_split",
            "n_obs",
            "C1_global_stability",
            "C2_mechanistic_stability",
            "C3_structural_stability",
            "ISI",
            "IG",
            "WG_risk",
            "VR_risk",
            "ER_mean_pnl",
            "TR_turnover",
        ]
        assert reader.fieldnames == expected_columns
        rows = list(reader)
        env_ids = {row["env_id"] for row in rows}
        assert "env_a" in env_ids and "env_b" in env_ids and "__overall__" in env_ids

    manifest_path = Path(run_ctx.output_dir) / "diagnostics_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    for key in ["seed", "git_hash", "config_hash", "instrument", "metric_basis", "isi_weights", "units", "created_utc"]:
        assert key in manifest
