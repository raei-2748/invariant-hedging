# HIRM - Hedging with Invariant Risk Minimization

[![CI](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml/badge.svg)](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository reproduces all results for *“Robust Generalization for Hedging under Crisis Regime Shifts.”* The release ships a
deterministic pipeline that stages data, trains the invariant hedging models, generates diagnostics, and exports the camera-ready
paper assets.

- 📄 Paper PDF: [Robust Generalization for Hedging under Crisis Regime Shifts](docs/paper.pdf)
- 📘 Reproduction playbook: [docs/REPRODUCE.md](docs/REPRODUCE.md)

## Quickstart

```bash
make data
make paper
make report-paper
```

The commands above download the SPY snapshot, execute the smoke-version of the paper harness (or the full sweep when `SMOKE=0`),
and compile the publication tables/figures under `reports/`.

## Repository layout

```
invariant-hedging/
├── src/
│   ├── core/            # Optimisation engine, solvers, and infrastructure
│   ├── modules/         # Data modules, environments, baselines, simulators
│   ├── evaluation/      # Crisis diagnostics, reporting, provenance utilities
│   ├── visualization/   # Plotting helpers used by the manuscript
│   └── legacy/          # Archived experiments kept for provenance only
├── experiments/         # Thin Hydra entrypoints for training & evaluation
├── configs/             # Versioned Hydra configs for experiments and reports
├── reports/             # Generated paper artefacts and analysis outputs
├── tests/               # Lightweight unit and smoke tests (`pytest -m "not heavy"`)
├── docs/                # Extended documentation and reproduction notes
└── meta/                # Provenance archives and paper changelog
```

### Paper → Code mapping

| Paper section | Code modules |
| --- | --- |
| §4 Invariance objectives | `src/core/losses.py`, `src/modules/head_invariance.py` |
| §5 Robust diagnostics | `src/modules/diagnostics.py`, `src/evaluation/analyze_diagnostics.py` |
| §6 Efficiency & crisis evaluation | `src/evaluation/evaluate_crisis.py`, `src/evaluation/reporting/` |
| Appendix (simulators & baselines) | `src/modules/sim/`, `src/modules/markets/`, `src/modules/baselines/` |

## Reproducibility & provenance

- `make paper` orchestrates the full suite and stores canonical outputs under `reports/paper_runs/` and `reports/paper_eval/`.
- `make report-paper` consumes those artefacts and assembles publication-ready CSV/figure bundles under `reports/paper/`.
- `meta/paper_provenance.json` is regenerated on every paper run and contains the resolved configs, git SHA, and runtime platform.
- Historical bundles prior to 1.0 live in `meta/archive_pre_v1/` for transparency.

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for hardware notes, runtime expectations, and troubleshooting tips.

## Continuous integration & tests

Smoke CI config (`.github/workflows/ci.yml`) exports `PYTHONPATH=src` and runs:

```bash
pytest -m "not heavy" --maxfail=1 --disable-warnings
make paper SMOKE=1
make report-paper
```

Local developers can mirror those checks with `make tests` or invoke `pytest -m "not heavy"` directly. Heavy calibration suites
are isolated behind `@pytest.mark.heavy` and excluded from CI.

## Data summary

The repository bundles a deterministic `data/spy_sample.csv` slice sufficient for smoke tests. Full experiments expect the staged
SPY options dataset retrieved via `make data`, which also snapshots metadata for later auditing.

## Citation

```
@software{wang2025robusthedging,
  author = {Ray Wang},
  title = {Robust Generalization for Hedging under Crisis Regime Shifts},
  year = {2025},
  url = {https://github.com/raei-2748/invariant-hedging}
}
```
