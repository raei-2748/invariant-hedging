from __future__ import annotations

import json
from pathlib import Path

import pytest

from invariant_hedging.reporting.legacy.schema import (
    FinalMetricsValidationError,
    load_final_metrics,
    validate_final_metrics,
)


def test_sample_final_metrics_file_validates() -> None:
    sample_path = Path("tests/data/report/final_metrics.json")
    payload = load_final_metrics(sample_path)
    assert "test/crisis_cvar" in payload.metrics
    assert payload.metrics["test/crisis_turnover"] == pytest.approx(10.0)


def test_nested_schema_round_trip(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0.0",
        "metrics": {
            "val/es95": {"value": -0.12, "units": "USD"},
            "val/turnover": {"value": 3.5},
        },
        "experiment": {"id": "exp_123", "tag": "smoke"},
        "notes": "synthetic validation sample",
    }
    path = tmp_path / "final_metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    validated = load_final_metrics(path)
    assert validated.schema_version == "1.0.0"
    assert validated.metrics["val/es95"] == pytest.approx(-0.12)
    assert validated.metadata["experiment"]["id"] == "exp_123"
    assert "notes" in validated.metadata


@pytest.mark.parametrize(
    "payload",
    [
        {"metrics": {"foo": "nan"}},
        {"metrics": {"foo": float("inf")}},
        {"foo": "bar"},
        {"metrics": {}},
        123,
    ],
)
def test_invalid_payloads_raise(payload: object) -> None:
    with pytest.raises(FinalMetricsValidationError):
        validate_final_metrics(payload)  # type: ignore[arg-type]
