from __future__ import annotations

import pandas as pd
import pytest

from src.report.aggregate import run_aggregation

pytest.importorskip("plotly")


def test_ire3d_outputs(sample_report_config):
    result = run_aggregation(sample_report_config, lite=False, skip_3d=False)

    tables_dir = result.outputs_dir / "tables"
    figures_dir = result.outputs_dir / "figures"
    interactive_dir = result.outputs_dir / "interactive"

    points_path = tables_dir / "ire_points.csv"
    assert points_path.exists()
    points = pd.read_csv(points_path)
    for axis in ["I_star", "R_star", "E_star"]:
        assert ((0.0 <= points[axis]) & (points[axis] <= 1.0)).all()

    metadata = result.provenance.get("ire3d", {})
    assert metadata
    for rho in metadata.get("spearman_rho", {}).values():
        assert rho < 0

    html_path = interactive_dir / "ire_3d.html"
    assert html_path.exists()
    assert "<html" in html_path.read_text().lower()

    for projection in sample_report_config["report"]["ire3d"]["projections"]:
        assert (figures_dir / f"ire_3d_{projection}.pdf").exists()
        assert (figures_dir / f"ire_3d_{projection}.png").exists()
