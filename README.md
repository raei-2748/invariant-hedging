# HIRM - Hedging with Invariant Risk Minimization

[![CI](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml/badge.svg)](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](reports/coverage/index.html)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Cite this work](https://img.shields.io/badge/citation-CITATION.cff-orange.svg)](CITATION.cff)

> Version 1.0.0 (Camera-Ready, October 2025)

This repository reproduces all results for *“Robust Generalization for Hedging under Crisis Regime Shifts.”* The release ships a
deterministic pipeline that stages data, trains the invariant hedging models, generates diagnostics, and exports the camera-ready
paper assets.

- Paper PDF: [Robust Generalization for Hedging under Crisis Regime Shifts](docs/paper.pdf)
- DOI landing page: [10.5281/zenodo.xxxxxx](https://doi.org/10.5281/zenodo.xxxxxx)
- Reproduction playbook: [docs/REPRODUCE.md](docs/REPRODUCE.md)

## Quickstart

```bash
make data
make paper
make report-paper
```

The commands above download the SPY snapshot, execute the smoke-version of the paper harness (or the full sweep when `SMOKE=0`),
and compile the publication tables/figures under `reports/`.

### Apple Silicon (MPS) acceleration

The training and evaluation pipelines auto-detect Apple Silicon GPUs via PyTorch's Metal Performance Shaders (MPS) backend. On
macOS, the launcher prints the resolved device plus CUDA/MPS availability before the first batch executes. If an MPS-capable
PyTorch build is missing, the run falls back to CPU and emits a warning describing how to enable GPU acceleration.

Install the PyTorch wheel that bundles the Metal backend inside your virtual environment (Apple ships the GPU kernels with the CPU
wheel):

```bash
python3 -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

Recreate the environment afterwards with `pip install -r requirements-lock.txt` to ensure every dependency matches the archival
release. When `torch.backends.mps.is_available()` reports `True`, running `make paper SMOKE=0` automatically selects the `mps`
device and disables mixed precision (AMP) for numerical stability. Expect a 2–3× speed-up relative to CPU-only runs on an M2
Pro/Max for the full 30-seed sweep, though individual ops may still fall back to CPU if the Metal kernels are unavailable.

> **Limitations.** PyTorch's MPS backend does not currently support `torch.cuda.amp`, so training always executes in `float32`.
> CUDA-based systems continue to operate unchanged and still deliver the fastest wall-clock reproduction.

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
└── archive/             # Provenance archives and paper changelog
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
- `archive/paper_provenance.json` is regenerated on every paper run and contains the resolved configs, git SHA, and runtime platform.
- Historical bundles prior to 1.0 live in `archive/v0.9/` for transparency.
- Release history is tracked in [CHANGELOG.md](CHANGELOG.md).

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

Please cite the accompanying paper and software release if you build upon this code base:

```
@software{wang2025robusthedging,
  author    = {Ray Wang},
  title     = {Robust Generalization for Hedging under Crisis Regime Shifts},
  version   = {1.0.0},
  year      = {2025},
  doi       = {10.5281/zenodo.xxxxxx},
  url       = {https://github.com/raei-2748/invariant-hedging}
}
```

The full citation metadata is mirrored in [`CITATION.cff`](CITATION.cff).
