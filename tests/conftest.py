import sys
from pathlib import Path
from typing import Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def float_tolerance() -> Dict[str, float]:
    """Numerical guard rails for deterministic torch tests.

    The diagnostics and penalty helpers under test ultimately reduce to
    operations such as averages, variances, and cosine similarities that are
    evaluated in ``float32``.  Analytic expectations are derived in ``float``
    (double precision) which means the comparisons can accumulate round-off
    discrepancies around the ``1e-7`` to ``1e-6`` range.  The returned
    tolerances therefore document the acceptable absolute/relative error bounds
    for equality checks against the closed-form references.
    """

    return {"abs": 1e-6, "rel": 1e-6}
