# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added figure-generation pipeline converting Track 4 tables into publication-ready plots, CLI entrypoints, and manifest logging.
- Documented plotting workflow and added CI coverage plus artifact export.
- Hardened the figure pipeline against schema drift with table-column aliases and defensive skipping for missing diagnostics.
- Added crisis stress toggles for synthetic simulation, including Heston+Merton jump overlay and volatility-linked liquidity costs with provenance outputs.
- Added deterministic real-market anchor loader with episode tagging and
  canonical output paths.
- Documented data expectations for real-market runs and shipped default
  `configs/examples/real_anchors.yaml` reference configuration.
- Implemented head-only HIRM gradient alignment with ψ-only penalties and
  optional φ freezing via `model.freeze_phi`.
- Added lightweight training loop and CLI (`tools.quick_run`) powered by
  `configs/examples/hirm_minimal.yaml` for rapid experiments and diagnostics.
- Stream alignment diagnostics to `alignment_head.csv` and documented the
  workflow in `docs/objectives.md` and `docs/devnotes/hirm_head.md`.

## v0.1.0 - 2025-10-04

- Baseline ERM-v1 configuration frozen as Phase 1 snapshot.
- Added reproducibility harness (locked requirements, metadata, smoke configs).
- Introduced CI pipeline with smoke training and reproducibility diff checks.
- Documented configs, contribution workflow, and Phase 1 metrics.

## Unreleased

- Implemented full diagnostic suite (ISI, IG, WG, VR, ER, TR) with unit tests.
- Added aggregation script to build consolidated tables for Track 5 figures.
- Documented diagnostic formulas and outputs in `docs/diagnostics.md`.
