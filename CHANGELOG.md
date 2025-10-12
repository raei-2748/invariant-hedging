# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added figure-generation pipeline converting Track 4 tables into publication-ready plots, CLI entrypoints, and manifest logging.
- Documented plotting workflow and added CI coverage plus artifact export.
- Hardened the figure pipeline against schema drift with table-column aliases and defensive skipping for missing diagnostics.

## v0.1.0 - 2025-10-04

- Baseline ERM-v1 configuration frozen as Phase 1 snapshot.
- Added reproducibility harness (locked requirements, metadata, smoke configs).
- Introduced CI pipeline with smoke training and reproducibility diff checks.
- Documented configs, contribution workflow, and Phase 1 metrics.
