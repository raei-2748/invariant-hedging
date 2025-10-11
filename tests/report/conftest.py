from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest


REGIMES = ("train_main", "crisis_2020")


def _make_seed_run(root: Path, seed: int) -> Path:
    run_dir = root / f"run_{seed:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for regime in REGIMES:
        base = 1.0 if regime == "train_main" else 1.2
        rows.append(
            {
                "regime": regime,
                "ISI": base + 0.1 * seed,
                "C1_global_stability": base - 0.05 * seed,
                "C2_mechanistic_stability": base + 0.02 * seed,
                "C3_structural_stability": base - 0.03 * seed,
                "IG": base * 0.5 + 0.05 * seed,
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / f"diagnostics_seed_{seed}.csv", index=False)

    final_metrics = {}
    for regime in REGIMES:
        shift = 0.1 if regime == "train_main" else 0.2
        final_metrics[regime] = {
            "WG_risk": 0.1 * (seed + 1) + shift,
            "VR_risk": 0.2 * (seed + 1) + shift,
            "CVaR_95": 0.3 * (seed + 1) + shift,
            "ER_mean_pnl": 0.5 * (seed + 1) - shift,
            "TR_turnover": 0.05 * (seed + 1) + 0.01 * seed,
        }
    (run_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (run_dir / "diagnostics_manifest.json").write_text(
        json.dumps({"seed": seed, "regimes": list(REGIMES)}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps({"seed": seed, "git": "deadbeef"}, indent=2),
        encoding="utf-8",
    )
    return run_dir


@pytest.fixture
def sample_report_config(tmp_path: Path) -> Dict[str, object]:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for seed in range(3):
        _make_seed_run(runs_root, seed)

    config = {
        "report": {
            "seeds": 30,
            "seed_dirs": [str(runs_root / "*")],
            "outputs_dir": str(tmp_path / "outputs"),
            "regimes_order": list(REGIMES),
            "confidence_level": 0.95,
            "include_gfc": True,
            "metrics": {
                "invariance": [
                    "ISI",
                    "C1_global_stability",
                    "C2_mechanistic_stability",
                    "C3_structural_stability",
                    "IG",
                ],
                "robustness": ["WG_risk", "VR_risk", "CVaR_95"],
                "efficiency": ["ER_mean_pnl", "TR_turnover"],
            },
            "qq": {"reference": "gaussian", "bins": 32},
            "figures": {"dpi_preview": 80, "backend": "agg", "qq_reference": "gaussian"},
            "latex": {"table_float": "t", "column_format": "lrr", "booktabs": True},
            "generate_3d": True,
            "ire3d": {
                "winsor_pct": [5.0, 95.0],
                "axis_I": "IG",
                "axis_R_source": "CVaR_95",
                "E_alpha_mode": "sd_equalize",
                "projections": ["top", "front", "side"],
            },
        }
    }
    return config
