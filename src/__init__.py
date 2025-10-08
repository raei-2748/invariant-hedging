"""Package initialisation for HIRM.

Applying conservative OpenMP defaults before PyTorch imports keeps smoke
tests functional inside restricted CI sandboxes where Intel's runtime cannot
create its shared-memory segment.
"""

from __future__ import annotations

import os

_OPENMP_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_THREADING_LAYER": "SEQUENTIAL",
    "KMP_AFFINITY": "disabled",
    "KMP_INIT_AT_FORK": "FALSE",
}


for _key, _value in _OPENMP_DEFAULTS.items():
    os.environ.setdefault(_key, _value)


__all__: list[str] = []
