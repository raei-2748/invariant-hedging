from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path

import matplotlib

from invariant_hedging import get_repo_root
from tests.figures.utils import make_run_directory

matplotlib.use("Agg")


def _load_make_all_figures():
    script_path = get_repo_root() / "tools/scripts/make_all_figures.py"
    spec = util.spec_from_file_location("tools.scripts.make_all_figures", script_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module


make_all_figures = _load_make_all_figures()


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
