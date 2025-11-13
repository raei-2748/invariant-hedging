from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from invariant_hedging.evaluation.reporting import paper_provenance


@pytest.fixture()
def dummy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "2023-01-01_000000"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("seed: 0\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.pt").write_bytes(b"checkpoint")
    return run_dir


def test_collect_provenance_includes_expected_fields(dummy_run: Path) -> None:
    payload = paper_provenance.collect_provenance(dummy_run)
    assert payload["git_hash"]
    assert payload["pip"] == sorted(payload["pip"])
    assert "python" in payload and "version" in payload["python"]
    run = payload["run"]
    assert run["exists"] is True
    files = run["files"]
    assert "config.yaml" in files
    config_digest = files["config.yaml"]["sha256"]
    hasher = hashlib.sha256()
    hasher.update((dummy_run / "config.yaml").read_bytes())
    assert config_digest == hasher.hexdigest()


def test_collect_provenance_handles_missing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing"
    payload = paper_provenance.collect_provenance(run_dir)
    assert payload["run"]["exists"] is False


def test_write_provenance_roundtrip(dummy_run: Path, tmp_path: Path) -> None:
    payload = paper_provenance.collect_provenance(dummy_run)
    output = tmp_path / "manifest.json"
    paper_provenance.write_provenance(output, payload)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded.keys() == payload.keys()
