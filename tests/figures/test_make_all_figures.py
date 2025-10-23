from __future__ import annotations

import json
from pathlib import Path

import matplotlib

from tools.scripts import make_all_figures
from tests.figures.utils import make_run_directory

matplotlib.use("Agg")


def test_make_all_figures_end_to_end(tmp_path: Path) -> None:
    run_dir = make_run_directory(tmp_path)
    make_all_figures.main(
        [
            "--run_dir",
            str(run_dir),
            "--dpi",
            "120",
            "--format",
            "png",
        ]
    )

    figures = {
        "fig_invariance_vs_ig.png",
        "fig_capital_efficiency_frontier.png",
        "fig_ire_scatter_3d.png",
        "fig_regime_panels.png",
        "fig_alignment_curves.png",
    }

    out_dir = run_dir / "figures"
    for name in figures:
        path = out_dir / name
        assert path.exists(), f"missing figure {name}"
        assert path.stat().st_size > 0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    manifest_names = {entry["name"] for entry in manifest}
    expected_manifest_names = {
        "fig_invariance_vs_ig",
        "fig_capital_efficiency_frontier",
        "fig_ire_scatter_3d",
        "fig_regime_panels",
        "fig_alignment_curves",
    }
    assert expected_manifest_names <= manifest_names

