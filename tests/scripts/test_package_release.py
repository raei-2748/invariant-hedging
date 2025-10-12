import json
import tarfile
from pathlib import Path

import pytest

from scripts.package_release import ReleaseConfig, package_release


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def sample_layout(tmp_path: Path) -> dict[str, Path]:
    env = tmp_path / "env.yml"
    _write(env, "dependencies: []\n")

    data_root = tmp_path / "data" / "sample"
    _write(data_root / "README.txt", "sample data")

    report_dir = tmp_path / "report"
    _write(report_dir / "tables" / "scorecard.csv", "method,metric,value\nerm,score,1.0\n")
    _write(report_dir / "figures" / "figure.png", "binary")
    _write(report_dir / "manifests" / "aggregate_manifest.json", "{}")

    provenance = tmp_path / "paper_provenance.json"
    _write(provenance, "{\n  \"dataset_snapshot\": \"unit-test\"\n}\n")

    metrics = tmp_path / "final_metrics.json"
    _write(metrics, "{\n  \"runs\": []\n}\n")

    return {
        "env": env,
        "data": data_root,
        "report": report_dir,
        "provenance": provenance,
        "metrics": metrics,
    }


def test_package_release(tmp_path: Path, sample_layout: dict[str, Path]) -> None:
    output_dir = tmp_path / "bundle"
    config = ReleaseConfig(
        tag="v0.9-paper-rc",
        output_dir=output_dir,
        report_dir=sample_layout["report"],
        environment_path=sample_layout["env"],
        data_root=sample_layout["data"],
        provenance_path=sample_layout["provenance"],
        metrics_path=sample_layout["metrics"],
        docker_digest="ghcr.io/example/image@sha256:deadbeef",
        golden_patterns=["*.csv"],
    )

    manifest_path = package_release(config)

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Environment artefacts recorded
    env_meta = manifest["environment"][sample_layout["env"].name]
    assert env_meta["path"] == f"environment/{sample_layout['env'].name}"
    assert env_meta["size"] > 0

    docker_meta = manifest["environment"]["docker_digest"]
    assert docker_meta["path"].endswith("docker_digest.txt")

    # Golden CSV copied and hashed
    golden_meta = manifest["golden"]["scorecard.csv"]
    assert golden_meta["path"] == "golden/scorecard.csv"
    golden_path = output_dir / golden_meta["path"]
    assert golden_path.exists()

    # Data tarball contains the miniature dataset
    data_tar = output_dir / manifest["data"]["data-mini.tar.gz"]["path"]
    assert data_tar.exists()
    with tarfile.open(data_tar, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("README.txt") for name in names)

    # Reports directory mirrored
    copied_scorecard = output_dir / "reports" / "tables" / "scorecard.csv"
    assert copied_scorecard.exists()

    # Provenance records present
    provenance_meta = manifest["provenance"][sample_layout["provenance"].name]
    assert provenance_meta["path"] == f"provenance/{sample_layout['provenance'].name}"
