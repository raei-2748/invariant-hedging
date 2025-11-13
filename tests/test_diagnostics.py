from __future__ import annotations

import numpy as np

from invariant_hedging.modules.diagnostics import safe_eval_metric


def test_safe_eval_metric_handles_missing_key() -> None:
    metrics = {"existing": np.array([1.0, 2.0])}
    resolved = safe_eval_metric(metrics.get, "missing", -1.0)
    assert resolved == -1.0


def test_safe_eval_metric_average_is_float() -> None:
    metrics = {"existing": np.array([1.0, 2.0, 3.0])}
    resolved = safe_eval_metric(lambda key: float(metrics[key].mean()), "existing")
    assert isinstance(resolved, float)
    assert np.isclose(resolved, 2.0)
