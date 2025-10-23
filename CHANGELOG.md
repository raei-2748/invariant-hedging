# Changelog

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

