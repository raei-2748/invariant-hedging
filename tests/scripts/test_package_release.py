import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from scripts.package_release import (
    MissingAssetError,
    ReleaseConfig,
    ValidationError,
    package_release,
)


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
    _write(report_dir / "tables" / "metrics.json", "{\n  \"score\": 1.0\n}\n")
    _write(report_dir / "figures" / "figure.png", "binary")
    _write(report_dir / "scratch" / "notes.log", "temporary debug output")
    _write(report_dir / "temp" / "skip.tmp", "ignored")
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


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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
        golden_patterns=["*.csv", "*.json"],
        exclude_patterns=["scratch/*.log", "temp/*"],
        workers=2,
        verbose=True,
    )

    manifest_path = package_release(config)

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Exclusions recorded and respected
    assert manifest["reports"]["excludes"] == ["scratch/*.log", "temp/*"]
    assert not (output_dir / "reports" / "scratch" / "notes.log").exists()
    assert not (output_dir / "reports" / "temp" / "skip.tmp").exists()

    # Environment artefacts recorded
    env_meta = manifest["environment"][sample_layout["env"].name]
    assert env_meta["path"] == f"environment/{sample_layout['env'].name}"
    assert env_meta["size"] > 0
    assert env_meta["sha256"] == _hash(output_dir / env_meta["path"])

    docker_meta = manifest["environment"]["docker_digest"]
    assert docker_meta["path"].endswith("docker_digest.txt")

    # Golden CSV copied and hashed
    golden_meta = manifest["golden"]["scorecard.csv"]
    assert golden_meta["path"] == "golden/scorecard.csv"
    golden_path = output_dir / golden_meta["path"]
    assert golden_path.exists()
    assert golden_meta["sha256"] == _hash(golden_path)

    # JSON golden copied as well
    json_meta = manifest["golden"]["metrics.json"]
    assert json_meta["path"] == "golden/metrics.json"

    # Data tarball contains the miniature dataset
    data_tar = output_dir / manifest["data"]["data-mini.tar.gz"]["path"]
    assert data_tar.exists()
    with tarfile.open(data_tar, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("README.txt") for name in names)
    assert manifest["data"]["data-mini.tar.gz"]["sha256"] == _hash(data_tar)

    # Reports directory mirrored
    copied_scorecard = output_dir / "reports" / "tables" / "scorecard.csv"
    assert copied_scorecard.exists()

    # Provenance records present
    provenance_meta = manifest["provenance"][sample_layout["provenance"].name]
    assert provenance_meta["path"] == f"provenance/{sample_layout['provenance'].name}"

    # Report manifest includes sha256 entries
    report_entries = manifest["reports"]["files"]
    assert any(entry["path"].endswith("tables/scorecard.csv") for entry in report_entries)
    for entry in report_entries:
        if entry["path"].endswith("tables/scorecard.csv"):
            report_path = output_dir / entry["path"]
            assert entry["sha256"] == _hash(report_path)


@pytest.mark.parametrize(
    "patterns,expected",
    [
        (["*.csv"], {"scorecard.csv"}),
        (["*.json"], {"metrics.json"}),
        (["*.csv", "*.json"], {"scorecard.csv", "metrics.json"}),
    ],
)
def test_golden_pattern_variants(
    tmp_path: Path, sample_layout: dict[str, Path], patterns: list[str], expected: set[str]
) -> None:
    output_dir = tmp_path / "bundle"
    config = ReleaseConfig(
        tag="v0.9-paper-rc",
        output_dir=output_dir,
        report_dir=sample_layout["report"],
        environment_path=sample_layout["env"],
        data_root=sample_layout["data"],
        golden_patterns=patterns,
    )

    manifest_path = package_release(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert set(manifest["golden"].keys()) == expected


def test_invalid_tag_raises(tmp_path: Path, sample_layout: dict[str, Path]) -> None:
    config = ReleaseConfig(
        tag="paper-rc",
        output_dir=tmp_path / "bundle",
        report_dir=sample_layout["report"],
        environment_path=sample_layout["env"],
        data_root=sample_layout["data"],
    )

    with pytest.raises(ValidationError):
        package_release(config)


def test_missing_data_root(tmp_path: Path, sample_layout: dict[str, Path]) -> None:
    config = ReleaseConfig(
        tag="v0.9-paper-rc",
        output_dir=tmp_path / "bundle",
        report_dir=sample_layout["report"],
        environment_path=sample_layout["env"],
        data_root=tmp_path / "does-not-exist",
    )

    with pytest.raises(MissingAssetError):
        package_release(config)


def test_reuse_existing_data_tar(tmp_path: Path, sample_layout: dict[str, Path]) -> None:
    existing_tar = tmp_path / "prebuilt.tar.gz"
    with tarfile.open(existing_tar, "w:gz") as archive:
        archive.add(sample_layout["data"], arcname="data-mini")

    output_dir = tmp_path / "bundle"
    config = ReleaseConfig(
        tag="v0.9-paper-rc",
        output_dir=output_dir,
        report_dir=sample_layout["report"],
        environment_path=sample_layout["env"],
        data_root=sample_layout["data"],
        existing_data_tar=existing_tar,
    )

    manifest_path = package_release(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    staged_tar = output_dir / manifest["data"]["data-mini.tar.gz"]["path"]

    assert staged_tar.read_bytes() == existing_tar.read_bytes()


def test_workers_validation(tmp_path: Path, sample_layout: dict[str, Path]) -> None:
    config = ReleaseConfig(
        tag="v0.9-paper-rc",
        output_dir=tmp_path / "bundle",
        report_dir=sample_layout["report"],
        environment_path=sample_layout["env"],
        data_root=sample_layout["data"],
        workers=0,
    )

    with pytest.raises(ValidationError):
        package_release(config)
