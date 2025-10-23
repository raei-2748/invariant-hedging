# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- No unreleased changes. Submit pull requests against `main` to begin the v1.1 cycle.

## v1.0.0 - 2025-10-24

- Finalised repository layout (`src/{core,modules,evaluation,visualization,legacy}` and archival material under `archive/`).
- Routed all experiment artefacts to `reports/` and ensured `make paper` + `make report-paper` produce camera-ready assets.
- Added lightweight regression tests for the data pipeline, diagnostics, HGCA penalty, and paper harness provenance.
- Introduced a streamlined CI workflow running `pytest -m "not heavy"`, `make paper SMOKE=1`, and `make report-paper` with `PYTHONPATH=src`.
- Published the camera-ready README, paper PDF placeholder, and updated provenance manifest defaults for the 1.0 release.

## v0.1.0 - 2025-10-04

- Baseline ERM-v1 configuration frozen as Phase 1 snapshot.
- Added reproducibility harness (locked requirements, metadata, smoke configs).
- Introduced CI pipeline with smoke training and reproducibility diff checks.
- Documented configs, contribution workflow, and Phase 1 metrics.
