# Changelog

## [Unreleased]

### Added
- Deterministic smoke audit (`make smoke-check`) that logs to `runs/test_smoke/` and fails CI when metrics diverge.
- Data integrity validation (`make data-check`) to guard OptionMetrics/IvyDB staging issues before long training jobs.
- Dockerfile + README quickstart covering the CPU-only smoke pipeline and data licensing guidance.
- Sanity harness (`make real`) that trains ERM/HIRM on the staged real dataset, runs diagnostics, and emits a PASS/FAIL verdict via `scripts/compare_sanity.py`.

### Changed
- Dependencies are now pinned via `pyproject.toml`/`requirements-lock.txt` with Python constrained to `>=3.11,<3.12`.
- CI enforces ruff linting, offline WANDB mode, deterministic smoke tests, and continues to run the paper smoke harness.

## [1.0.0] - 2025-10-24

### Added
- Archive layout with `archive/` provenance bundle and v0.9 snapshot.
- Comprehensive reproduction guide covering every figure and table.
- Coverage reporting in CI and README badges, including Zenodo DOI metadata.

### Fixed
- Updated scripts and tests to track provenance under the new archive path.
- Normalised dependency versions and CI workflow to remove drift across machines.

### Reproducibility
- Locked Python dependencies in `requirements.txt`, `environment.yml`, and `requirements-lock.txt`.
- Documented deterministic seeds, runtime expectations, and hardware prerequisites in `docs/REPRODUCE.md`.
- Added make targets and CI hooks that rely on the lock file for consistent builds.
