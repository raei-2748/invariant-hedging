from __future__ import annotations

import json
from pathlib import Path

import matplotlib

from invariant_hedging.reporting.plot_io import parse_formats
from invariant_hedging.visualization import (
    plot_alignment_curves,
    plot_capital_efficiency_frontier,
    plot_invariance_vs_ig,
    plot_ire_scatter_3d,
    plot_regime_panels,
)
from tests.figures.utils import make_run_directory

matplotlib.use("Agg")


def test_individual_plot_scripts(tmp_path: Path) -> None:
    run_dir = make_run_directory(tmp_path)
    out_dir = run_dir / "figures"

    formats = parse_formats("png,pdf")

    assert plot_invariance_vs_ig.create_figure(
        run_dir=run_dir,
        out_dir=out_dir,
        formats=formats,
        dpi=150,
        style="journal",
    )
    assert (out_dir / "fig_invariance_vs_ig.png").stat().st_size > 0

    assert plot_capital_efficiency_frontier.create_figure(
        run_dir=run_dir,
        out_dir=out_dir,
        formats=formats,
        dpi=150,
        style="journal",
    )
    assert (out_dir / "fig_capital_efficiency_frontier.pdf").stat().st_size > 0

    assert plot_ire_scatter_3d.create_figure(
        run_dir=run_dir,
        out_dir=out_dir,
        formats=formats,
        dpi=120,
        style="journal",
        separate_by_regime=True,
    )
    assert (out_dir / "fig_ire_scatter_3d.png").exists()
    assert (out_dir / "fig_ire_scatter_3d_byregime.pdf").exists()

    assert plot_regime_panels.create_figure(
        run_dir=run_dir,
        out_dir=out_dir,
        formats=("png",),
        dpi=150,
        style="journal",
    )
    assert (out_dir / "fig_regime_panels.png").exists()

    assert plot_alignment_curves.create_figure(
        run_dir=run_dir,
        out_dir=out_dir,
        formats=("pdf",),
        dpi=150,
        style="journal",
    )
    assert (out_dir / "fig_alignment_curves.pdf").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    names = {entry["name"] for entry in manifest}
    assert "fig_invariance_vs_ig" in names
    assert "fig_alignment_curves" in names

