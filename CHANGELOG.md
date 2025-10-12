# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added deterministic real-market anchor loader with episode tagging and
  canonical output paths.
- Documented data expectations for real-market runs and shipped default
  `config/real_anchors.yaml` reference configuration.

## v0.1.0 - 2025-10-04

- Baseline ERM-v1 configuration frozen as Phase 1 snapshot.
- Added reproducibility harness (locked requirements, metadata, smoke configs).
- Introduced CI pipeline with smoke training and reproducibility diff checks.
- Documented configs, contribution workflow, and Phase 1 metrics.
