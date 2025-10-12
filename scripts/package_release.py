#!/usr/bin/env python3
"""Bundle the assets required for a tagged paper release.

The packaging workflow copies the reproducibility primitives referenced in the
paper (environment, minimal dataset, golden tables, provenance manifests and
the rendered reports) into a deterministic directory tree.  The tree can then
be zipped and attached to a GitHub Release.

The script intentionally avoids any heavyweight dependencies so it can run from
the same Python environment used for training.  It emits a `manifest.json`
describing every file that was copied along with SHA256 checksums.  This
manifest is later consumed by CI and downstream reproducibility checks.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, MutableMapping


DEFAULT_REPORT_DIR = Path("outputs") / "report_paper"
DEFAULT_PROVENANCE = Path("paper_provenance.json")
DEFAULT_METRICS = Path("runs") / "paper" / "final_metrics.json"
DEFAULT_DATA_ROOT = Path("data") / "sample"
DEFAULT_ENVIRONMENT = Path("environment.yml")


class PackagingError(RuntimeError):
    """Raised when required artefacts are missing or the bundle fails."""


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_asset(manifest: MutableMapping[str, object], key: str, path: Path, *, root: Path) -> None:
    manifest[key] = {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": _hash_file(path),
    }


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise PackagingError(f"{description} not found at {path}")


def _docker_digest_from_image(image: str) -> str:
    try:
        result = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{index .RepoDigests 0}}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on docker availability
        raise PackagingError("Docker CLI not available to resolve image digest") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on docker availability
        raise PackagingError(f"Failed to inspect docker image '{image}': {exc.stderr.strip()}".strip()) from exc
    digest = result.stdout.strip()
    if not digest:
        raise PackagingError(f"Docker image '{image}' did not report a RepoDigest")
    return digest


@dataclass
class ReleaseConfig:
    tag: str
    output_dir: Path
    report_dir: Path
    environment_path: Path
    data_root: Path
    provenance_path: Path | None = None
    metrics_path: Path | None = None
    docker_digest: str | None = None
    docker_image: str | None = None
    golden_patterns: Iterable[str] = ()


def _resolve_docker_digest(config: ReleaseConfig) -> str | None:
    if config.docker_digest:
        return config.docker_digest
    if config.docker_image:
        return _docker_digest_from_image(config.docker_image)
    env_digest = os.environ.get("PAPER_RELEASE_DOCKER_DIGEST")
    if env_digest:
        return env_digest.strip()
    return None


def _tar_data(source: Path, dest: Path) -> None:
    if dest.exists():
        dest.unlink()
    with tarfile.open(dest, "w:gz") as archive:
        archive.add(source, arcname="data-mini")


def package_release(config: ReleaseConfig) -> Path:
    report_dir = config.report_dir.resolve()
    environment_path = config.environment_path.resolve()
    data_root = config.data_root.resolve()
    output_dir = config.output_dir.resolve()
    provenance_path = config.provenance_path.resolve() if config.provenance_path else None
    metrics_path = config.metrics_path.resolve() if config.metrics_path else None

    _ensure_exists(report_dir, "Report artefacts directory")
    _ensure_exists(environment_path, "Environment specification")
    _ensure_exists(data_root, "Sample data root")
    if provenance_path is not None:
        _ensure_exists(provenance_path, "Paper provenance manifest")
    if metrics_path is not None:
        _ensure_exists(metrics_path, "Aggregate metrics file")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "tag": config.tag,
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "environment": {},
        "data": {},
        "reports": {
            "source": str(report_dir),
            "files": [],
        },
        "golden": {},
        "provenance": {},
    }

    environment_dir = output_dir / "environment"
    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    golden_dir = output_dir / "golden"
    provenance_dir = output_dir / "provenance"

    # Environment specification
    env_dest = environment_dir / environment_path.name
    _copy_file(environment_path, env_dest)
    _record_asset(manifest["environment"], environment_path.name, env_dest, root=output_dir)

    docker_digest = _resolve_docker_digest(config)
    if docker_digest:
        docker_path = environment_dir / "docker_digest.txt"
        _write_text(docker_path, docker_digest + "\n")
        _record_asset(manifest["environment"], "docker_digest", docker_path, root=output_dir)

    # Data tarball
    data_dir.mkdir(parents=True, exist_ok=True)
    data_tar = data_dir / "data-mini.tar.gz"
    _tar_data(data_root, data_tar)
    _record_asset(manifest["data"], "data-mini.tar.gz", data_tar, root=output_dir)
    checksum_path = data_dir / "data-mini.sha256"
    checksum = _hash_file(data_tar)
    _write_text(checksum_path, f"{checksum}  {data_tar.name}\n")
    _record_asset(manifest["data"], "data-mini.sha256", checksum_path, root=output_dir)

    # Reports directory (tables, figures, manifests, etc.)
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    shutil.copytree(report_dir, reports_dir)
    report_files: List[dict[str, object]] = []
    for path in sorted(reports_dir.rglob("*")):
        if not path.is_file():
            continue
        report_files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "size": path.stat().st_size,
            }
        )
    manifest["reports"]["files"] = report_files

    # Golden CSVs - copy selected tables for quick access
    golden_patterns = list(config.golden_patterns)
    tables_dir = reports_dir / "tables"
    if tables_dir.exists() and not golden_patterns:
        golden_patterns = ["*.csv"]
    for pattern in golden_patterns:
        for candidate in sorted(report_dir.glob(pattern)):
            if candidate.is_file():
                rel = candidate.name
                dest = golden_dir / rel
                _copy_file(candidate, dest)
                _record_asset(manifest["golden"], rel, dest, root=output_dir)
        if (tables_dir).exists():
            for candidate in sorted(tables_dir.glob(pattern)):
                if candidate.is_file():
                    rel = candidate.name
                    dest = golden_dir / rel
                    if dest.exists():
                        continue
                    _copy_file(candidate, dest)
                    _record_asset(manifest["golden"], rel, dest, root=output_dir)

    # Provenance manifests
    if provenance_path is not None:
        dest = provenance_dir / provenance_path.name
        _copy_file(provenance_path, dest)
        _record_asset(manifest["provenance"], provenance_path.name, dest, root=output_dir)
    if metrics_path is not None and metrics_path.exists():
        dest = provenance_dir / metrics_path.name
        _copy_file(metrics_path, dest)
        _record_asset(manifest["provenance"], metrics_path.name, dest, root=output_dir)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag (e.g. v0.9-paper-rc)")
    parser.add_argument("--output", help="Destination directory for the bundle", default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory containing rendered reports")
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT, help="environment.yml path")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory for the miniature dataset")
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE, help="paper_provenance.json path")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="Optional final_metrics.json path to include in provenance",
    )
    parser.add_argument(
        "--docker-image",
        help="Docker image reference to inspect for a digest (optional if --docker-digest provided)",
    )
    parser.add_argument(
        "--docker-digest",
        help="Explicit docker image digest (e.g. ghcr.io/org/image@sha256:...)",
    )
    parser.add_argument(
        "--golden",
        action="append",
        default=[],
        help="Glob pattern(s) for golden CSVs relative to the report directory",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output) if args.output else Path("releases") / args.tag
    config = ReleaseConfig(
        tag=args.tag,
        output_dir=output_dir,
        report_dir=args.report_dir,
        environment_path=args.environment,
        data_root=args.data_root,
        provenance_path=args.provenance,
        metrics_path=args.metrics if args.metrics else None,
        docker_digest=args.docker_digest,
        docker_image=args.docker_image,
        golden_patterns=args.golden,
    )
    try:
        manifest_path = package_release(config)
    except PackagingError as exc:
        raise SystemExit(str(exc))
    print(f"Release assets written to {config.output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
