# Phase 1 ERM Base Snapshot

This folder collects reproducible artefacts for the Phase 1 ERM baseline runs:

- `ERM_base_crisis.csv` — aggregate table emitted by `scripts/run_baseline.py`.
- Optional per-seed evaluation outputs when running with `--keep-eval`.
- Any derived reports or plots (e.g. bootstrap confidence intervals, crisis charts).

Use `git tag baseline-erm-base` once the snapshot is finalised so downstream experiments can reference it.
