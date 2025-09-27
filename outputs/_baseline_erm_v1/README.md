# Phase 1 ERM Baseline Snapshot

This folder collects reproducible artefacts for the Phase 1 ERM baseline runs:

- `summary.csv` — aggregate table emitted by `scripts/run_baseline.py`.
- Optional per-seed evaluation outputs when running with `--keep-eval`.
- Any derived reports or plots (e.g. bootstrap confidence intervals, crisis charts).

Use `git tag baseline-erm-v1` once the snapshot is finalised so downstream experiments can reference it.
