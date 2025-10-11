from __future__ import annotations

import re

from src.report.aggregate import run_aggregation


def test_latex_tables_format(sample_report_config):
    result = run_aggregation(sample_report_config, lite=True, skip_3d=True)
    tables_dir = result.outputs_dir / "tables"
    main_tex = (tables_dir / "main_scorecard.tex").read_text()

    assert "\\toprule" in main_tex
    assert "\\midrule" in main_tex
    assert "\\bottomrule" in main_tex

    header_line = next(line for line in main_tex.splitlines() if line.startswith("Metric &"))
    # Metric column plus each regime column should appear in header.
    expected_columns = len(sample_report_config["report"]["regimes_order"]) + 1
    assert header_line.count("&") == expected_columns - 1

    assert "C1\\_global\\_stability" in main_tex

    robustness_tex = (tables_dir / "robustness_table.tex").read_text()
    assert re.search(r"\\caption\{Robustness metrics\}", robustness_tex)
    assert "WG_risk" in (tables_dir / "robustness_table.csv").read_text()
