"""Schema validation for diagnostics long-format exports."""

import csv
import json
from pathlib import Path

import pandas as pd
import torch

from src.diagnostics.export import DiagnosticsRunContext, gather_and_export
from src.report.build_scorecard import build_scorecard


def _make_batch(risk, outcome, positions, grad, representation):
    return {
        "risk": torch.as_tensor(risk, dtype=torch.float32),
        "outcome": torch.as_tensor(outcome, dtype=torch.float32),
        "positions": torch.as_tensor(positions, dtype=torch.float32),
        "grad": torch.as_tensor(grad, dtype=torch.float32),
        "representation": torch.as_tensor(representation, dtype=torch.float32),
    }


def test_export_long_format_schema_and_idempotence(tmp_path):
    run_dir = tmp_path / "runs" / "example_run"
    run_ctx = DiagnosticsRunContext(
        output_dir=run_dir,
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
        run_id="example_run",
    )

    probe_cfg = {
        "batches": {
            "env_a": [
                _make_batch(
                    [0.1, 0.2],
                    [0.05, 0.06],
                    [[0.0, 0.1], [0.1, 0.1]],
                    [1.0, 0.0],
                    [[0.0, 0.0], [0.0, 0.0]],
                )
            ],
            "env_b": [
                _make_batch(
                    [0.3, 0.4],
                    [0.02, 0.01],
                    [[0.0, -0.1], [0.0, -0.1]],
                    [-1.0, 0.0],
                    [[1.0, 0.0], [1.0, 0.0]],
                )
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

    assert csv_path == run_dir / "scorecard.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_columns = ["run_id", "phase", "env", "metric", "value"]
        assert reader.fieldnames == expected_columns
        rows = list(reader)

    env_metrics = {(row["env"], row["metric"]) for row in rows}
    assert ("env_a", "ER_mean_pnl") in env_metrics
    assert ("env_b", "TR_turnover") in env_metrics
    assert ("__overall__", "ISI") in env_metrics

    # Idempotence: running export again should not change the CSV contents.
    before = csv_path.read_text(encoding="utf-8")
    second_path = gather_and_export(
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
    assert second_path == csv_path
    after = csv_path.read_text(encoding="utf-8")
    assert before == after

    manifest_path = Path(run_ctx.output_dir) / "diagnostics_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    for key in [
        "run_id",
        "phase",
        "seed",
        "git_hash",
        "config_hash",
        "instrument",
        "metric_basis",
        "isi_weights",
        "units",
        "created_utc",
    ]:
        assert key in manifest

    table = build_scorecard(pd.read_csv(csv_path))
    assert not table.empty
    assert "env_a__ER_mean_pnl" in table.columns
    assert "__overall____TR_turnover" in table.columns
    assert table.iloc[0]["run_id"] == "example_run"
    assert table.iloc[0]["phase"] == "eval"
