from __future__ import annotations

from pathlib import Path


def test_run_of_record_writes_to_reports_tree() -> None:
    script = Path("tools/run_of_record.sh").read_text(encoding="utf-8")
    assert "reports/paper_runs" in script
    assert "reports/paper_eval" in script
    assert "archive/paper_provenance.json" in script


def test_paper_provenance_placeholder_exists() -> None:
    provenance = Path("archive/paper_provenance.json")
    assert provenance.exists()
    contents = provenance.read_text(encoding="utf-8")
    assert "run_root" in contents
