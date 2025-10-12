import pandas as pd
import pytest

from src.diagnostics.schema import (
    CANONICAL_COLUMNS,
    normalize_diagnostics_frame,
    validate_diagnostics_table,
)


def test_validate_accepts_normalized_frame():
    data = {
        "env": ["train_main", "crisis_2020"],
        "split": ["test", "test"],
        "seed": [1, 1],
        "algo": ["erm", "erm"],
        "metric": ["ISI", "TR_turnover"],
        "value": [0.8, 1.2],
    }
    frame = pd.DataFrame(data, columns=CANONICAL_COLUMNS)
    normalized = normalize_diagnostics_frame(frame)
    validate_diagnostics_table(normalized)


def test_validate_raises_on_missing_column():
    frame = pd.DataFrame({"env": ["foo"]})
    with pytest.raises(ValueError):
        validate_diagnostics_table(frame)


def test_validate_raises_on_wrong_dtype():
    frame = pd.DataFrame(
        {
            "env": ["train"],
            "split": ["test"],
            "seed": ["not_an_int"],
            "algo": ["erm"],
            "metric": ["ISI"],
            "value": ["nan"],
        }
    )
    with pytest.raises(TypeError):
        validate_diagnostics_table(frame)
