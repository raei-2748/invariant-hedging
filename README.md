# HIRM
[![CI](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml/badge.svg)](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml) [![Python 3.10–3.11](https://img.shields.io/badge/python-3.10--3.11-blue.svg)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
This research introduce **HIRM** (Hedging with IRM), a research framework for robust hedging, designed to test whether Invariant Risk Minimization improves tail-risk performance compared to standard deep hedging approaches.

**Research question:** Does adding Invariant Risk Minimization (IRM) to a deep hedging framework improve robustness against regime shifts (such as crisis periods like Volmageddon), compared to a standard deep hedger without IRM or trained on alternative regularizations?

## Contents

- [Quick start](#quick-start)
- [Experiments](#phase-1-results) → [Phase 1](#phase-1-results) | [Phase 2 — Head-Only IRM + Diagnostics (Current)](#-phase-2--head-only-irm--diagnostics-current)
## Table of Contents
- [Quick start](#quick-start)
- [Roadmap](#roadmap)
  - [✅ Phase 1 — ERM Baseline (Complete)](#phase-1-erm)
  - [📈 Phase 2 — Head-Only IRM + Diagnostics (Current)](#phase-2-headirm)
- [Repository layout](#repository-layout)
- [Logging and outputs](#logging-and-outputs)
- [Testing](#testing)
- [Smoke test](#smoke-test)
- [Data](#data)
- [Reproducibility](#reproducibility)
- [Reproducibility checklist](#reproducibility-checklist)

## Quick start

## Phase 1 results

A frozen snapshot of the Phase 1 hedging runs is available under [`outputs/_phase1_snapshot/`](outputs/_phase1_snapshot/). The crisis metrics below mirror `outputs/_phase1_snapshot/final_metrics.json` and can be regenerated with:

```bash
make reproduce PHASE=phase1
```

| Model    | Crisis CVaR-95 | Mean PnL | Turnover |
|----------|----------------|----------|----------|
| ERM-v1   | –12.4%         | 0.021    | 1.00×    |
| IRM-head | –10.7%         | 0.020    | 1.12×    |
| V-REx    | –11.0%         | 0.019    | 1.05×    |


### 📈 Phase 2 — Head-Only IRM + Diagnostics (Current)

Phase 2 expands the baseline to test **head-only IRM**, **V-REx**, and early diagnostics (**IG**, **WG**, **MSI**) under jump + liquidity stresses.  See the full plan in [`experiments/phase2_plan.md`](experiments/phase2_plan.md).

**Quick Start**

```bash
# IRM-head sweep (λ in {1e-2, 1e-1, 1})
python scripts/train.py config=train/phase2 irm.enabled=true irm.mode=head_only irm.lambda=0.1

# V-REx sweep (β in {1, 5, 10})
python scripts/train.py config=train/phase2 vrex.enabled=true vrex.beta=10.0
```

Results are saved in `outputs/_phase2_snapshot/`.


1. **Install dependencies**
   ```bash
   make setup
   ```
2. **Run a training job** (defaults to the ERM baseline)
   ```bash
   make train
   ```
   Override the Hydra config by passing `CONFIG=train/irm` or other configs to the make target.
3. **Evaluate a checkpoint** on the crisis out-of-distribution regime
   ```bash
   make evaluate CHECKPOINT=/path/to/checkpoint.pt
   ```
4. **Full reproduction** of the Phase 1 protocol (ERM, ERM-reg, IRM, GroupDRO, V-REx)
   ```bash
   scripts/make_reproduce.sh
   ```
   Each run mirrors metrics locally under `runs/<timestamp>/` and, if W&B is available, logs to the `invariant-hedging` project.

## Roadmap

<a id="phase-1-erm"></a>
### Phase 1 — ERM Baseline (Complete)
See [`experiments/phase1_summary.md`](experiments/phase1_summary.md) for the ERM-v1 baseline protocol and artifacts.

<a id="phase-2-headirm"></a>
### Phase 2 — Head-Only IRM + Diagnostics (Current)
See [`experiments/phase2_plan.md`](experiments/phase2_plan.md) for objectives, sweeps, and metrics.

## Repository layout

```
configs/               Hydra configs for data, environments, models, training and evaluation
src/
  data/                Synthetic generators, real-data anchor, feature engineering
  markets/             Black–Scholes pricing and execution cost models
  envs/                Daily rebalancing single-asset environment
  objectives/          CVaR estimator, entropic risk and regularisation penalties
  models/              Policy networks and representation heads
  utils/               Logging, checkpoints, determinism helpers, statistics
  train.py             Three-phase training loop (ERM → IRM ramp → full horizon)
  eval.py              Crisis evaluation with CVaR-95 table and QQ plots
scripts/               Convenience wrappers (`run_train.sh`, `run_eval.sh`, `make_reproduce.sh`)
tests/                 Unit tests for pricing, CVaR, costs and seeding
notebooks/             Diagnostics (e.g. `phi_invariance.ipynb`)
```

## Logging and outputs

Every run (train or eval) writes to `runs/<timestamp>/` with the following structure:

- `config.yaml`: resolved Hydra configuration
- `metrics.jsonl`: stepwise metrics
- `final_metrics.json`: summary metrics
- `checkpoints/`: top-k checkpoints selected by validation CVaR-95
- `artifacts/`: crisis tables, QQ plots and any additional evaluation artefacts
- `metadata.json`: git commit hash, platform fingerprint, Python and PyTorch versions

If W&B credentials are available the same metrics are mirrored to the `invariant-hedging` project; offline mode is supported via `WANDB_MODE=offline`.

### Reports

Aggregate tables and publication-ready figures can be regenerated locally with:

```bash
make report
```

This command scans the latest per-seed CSV exports, renders the Phase-2 diagnostics (penalty sweeps, ablations, ISI decomposition, cross-regime heatmaps, and existing scorecard charts), and writes all artefacts to [`outputs/report_assets/`](outputs/report_assets/). The legacy `make phase2_scorecard` entry point now forwards to `make report` for backward compatibility.

## Testing

Unit tests cover pricing Greeks, CVaR estimation, cost kernels and deterministic seeding. Run them with:

```bash
make tests
```

## Smoke test

The helper scripts export conservative Intel OpenMP settings so PyTorch runs even in sandboxes with no shared-memory segment. Kick off the short training loop with:

```bash
scripts/run_train.sh train/smoke
```

If you prefer to invoke Python directly, mirror those defaults explicitly:

```bash
OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE python3 -m src.train --config-name=train/smoke
```

GitHub Actions executes the test suite together with a smoke train/eval pass on every push.

## Data

The synthetic generator supports GBM and Heston dynamics with environment-specific transaction costs. A tiny SPY options slice (`data/spy_sample.csv`) is bundled as a deterministic real-data anchor that exercises the full feature pipeline.

### SPY real-data splits (paper-aligned)

Paper experiments reference fixed SPY windows for training, validation, and crisis tests. The repository now encodes those ranges as versioned YAML under [`configs/splits/`](configs/splits/):

- [`spy_train.yaml`](configs/splits/spy_train.yaml) — train on low/medium volatility from **2017-01-03 → 2019-12-31**.
- [`spy_val.yaml`](configs/splits/spy_val.yaml) — validate on the **2018 volatility spike** (default `2018-10-01 → 2018-12-31`). The paper leaves the exact endpoints unspecified; this default is **paper-unspecified—settable via YAML** so authors can refine the window without code changes.
- Crisis tests [`spy_test_2018.yaml`](configs/splits/spy_test_2018.yaml), [`spy_test_2020.yaml`](configs/splits/spy_test_2020.yaml), [`spy_test_2022.yaml`](configs/splits/spy_test_2022.yaml) capture Volmageddon, COVID, and inflation/tightening regimes, respectively.
- [`spy_test_2008.yaml`](configs/splits/spy_test_2008.yaml) adds a held-out **GFC anchor** for extended stress testing.

Generate a slice with:

```bash
python -m src.data.spy_loader --split configs/splits/spy_train.yaml --out_parquet outputs/slices/spy_train.parquet --runs_dir runs/spy_train
```

Each invocation prints the covered date span, writes an optional Parquet file, and records provenance (split YAML, git hash, Python/platform fingerprint) under `runs/<timestamp>/metadata.json`. Because the splits live in YAML, downstream papers should update and version-control any boundary tweaks alongside their experiment logs.

Episode configuration, cost files and model settings live under `configs/`. Adjust these as needed for experiments or sweeps. The default training protocol performs 20k ERM pre-training steps, a 10k IRM ramp, and continues until 150k total updates with environment-balanced batching.

## Reproducibility

`scripts/make_reproduce.sh` re-runs the ERM, ERM-reg, IRM, GroupDRO and V-REx configurations for seed 0, evaluates the best checkpoint for each on the crisis environment, and regenerates the crisis CVaR-95 table plus QQ plots. All seeds are controlled via `configs/train/*.yaml` and `src/utils/seed.py` to guarantee deterministic `metrics.jsonl` for `seed=0`.

## Reproducibility checklist

- Deterministic seeds for training, evaluation, and tests (`seed_list.txt` and Hydra configs).
- Resolved Hydra configs saved under each run directory (`runs/<timestamp>/config.yaml`).
- Metrics logged per-step (`metrics.jsonl`) and in aggregate (`final_metrics.json`) including CVaR-95, Sharpe, and turnover.
- Run metadata captured in `metadata.json` with git commit, platform, Python, and PyTorch versions.
