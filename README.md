# HIRM

[![CI](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml/badge.svg)](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml)
[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10--3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper reproduction guide](https://img.shields.io/badge/reproduce-paper-blue.svg)](REPRODUCE.md)

HIRM (Hedging with IRM) is a research codebase for reproducing the paper results on
robust hedging under regime shifts. This repository now defaults to the
**paper reproduction workflow**: acquiring the SPY dataset snapshot, running the
compact training/evaluation pipeline, and generating the camera-ready tables and
figures.

For a full walkthrough of every command, expected artefacts, and provenance
requirements see [REPRODUCE.md](REPRODUCE.md).

## Quickstart

1. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```
2. **Stage the SPY dataset snapshot**
   ```bash
   make data
   ```
3. **Preview the end-to-end workflow** (prints the commands that will run)
   ```bash
   scripts/run_of_record.sh --dry-run
   ```
4. **Train + evaluate the paper configuration on CPU**
   ```bash
   make paper
   ```
   Each run mirrors metrics locally under `runs/<timestamp>/`. Enable remote logging with `logger.use_wandb=true` (defaults to
   `false`) to stream metrics to the `invariant-hedging` project when credentials are available.

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

Set `logger.use_wandb=false` (the default) to keep runs completely local while still writing identical `final_metrics.json`, `metadata.json`, and artefacts under `runs/<timestamp>/`. Use `logger.use_wandb=true` to stream to W&B alongside the local mirror.

### PR-05 Reporting

The reporting pipeline aggregates cross-seed diagnostics, emits publication-ready tables/figures, and captures provenance in [`outputs/report_assets/`](outputs/report_assets/). Use the convenience targets:

```bash
make report          # full 30-seed aggregation with 3D I–R–E assets
make report-lite     # ≤5 seeds, skips heavy plots for CI runs
```

Both targets resolve [`configs/report/default.yaml`](configs/report/default.yaml). Adjust metric blocks, QQ options, and the 3D toggle there. Set `generate_3d: false` or pass `--skip-3d` via `scripts/aggregate.py` to disable the I–R–E projection entirely. Outputs include LaTeX tables, CSV mirrors, heatmaps, QQ plots, seed distributions, efficiency frontiers, and (optionally) interactive + static I–R–E visualisations.

## Testing

Unit tests cover pricing Greeks, CVaR estimation, cost kernels and deterministic seeding. Run them with:

```bash
make tests
```

## Diagnostics (PR-04): Invariance–Robustness–Efficiency

The diagnostics stack exports per-seed scorecards that align with the paper’s
I–R–E geometry. After evaluation each enabled run writes a tidy CSV and a JSON
manifest under `runs/<timestamp>/` (or a custom directory configured via
`diagnostics.outputs.dir`). CSV rows contain one entry per environment plus an
`__overall__` aggregate with the following columns:

| Column | Description |
| --- | --- |
| `seed, git_hash, exp_id` | Run provenance and experiment identifiers. |
| `split_name, regime_tag, env_id, is_eval_split` | Split metadata and environment labels. |
| `n_obs` | Number of observations used for the diagnostic probe. |
| `C1_global_stability` | Normalised risk dispersion across environments (1 = stable). |
| `C2_mechanistic_stability` | Cosine-alignment of head gradients across environments. |
| `C3_structural_stability` | Representation similarity across environments. |
| `ISI` | Weighted combination of C1/C2/C3 clipped to `[0, 1]`. |
| `IG` | Invariance gap (dispersion of realised outcomes). |
| `WG_risk`, `VR_risk` | Worst-group risk and variance across environments. |
| `ER_mean_pnl`, `TR_turnover` | Efficiency metrics: mean PnL and turnover. |

The manifest (`diagnostics_manifest.json`) records the seed, git hash,
config hash, instrument, metric basis, ISI weights, units, and creation
timestamp for reproducibility.

### Running the diagnostics probe

1. Enable diagnostics in the Hydra config (see `configs/diagnostics/default.yaml`):

   ```yaml
   defaults:
     - diagnostics: default
5. **Build publication tables/figures from the paper run**
   ```bash
   make report-paper
   ```
6. *(Optional)* **Rebuild the full report assets for multi-seed runs**
   ```bash
   make report
   ```

2. Provide diagnostic batches (held-out from training) via
   `diagnostics.probe`. Batches are dictionaries containing `risk`, `outcome`,
   `positions`, and optionally `grad`/`representation` tensors per environment.

3. Run evaluation. The export helper will assemble I–R–E metrics using the new
   modules in `src/diagnostics/` and log the output CSV path via the existing
   run logger.

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

## Simulation Regimes & Calibration (PR-02)

PR-02 introduces YAML-driven calibration hooks so synthetic markets can flip between calm and crisis regimes without touching code. The configs live under `configs/sim`:

- `heston_*.yaml` toggle the base diffusion (mean reversion, vol-of-vol, correlations) while pinning the random seed that flows into `SimRecipe`.
- `merton_*.yaml` overlay Merton jump diffusion (λ, μ<sub>j</sub>, σ<sub>j</sub>) and can be omitted to disable jumps.
- `liquidity_*.yaml` widen transaction costs through a variance-linked spread multiplier.
- `sabr_*.yaml` capture lognormal SABR smiles for pricing sanity checks.

Use `src.sim.calibrators.compose_sim_recipe` to merge a base diffusion with optional jump and liquidity overlays:

```python
from src.sim.calibrators import compose_sim_recipe

calm_recipe = compose_sim_recipe(
    "heston",
    "configs/sim/heston_calm.yaml",
    "configs/sim/merton_calm.yaml",
    "configs/sim/liquidity_calm.yaml",
    seed=None,
)
```

Swapping the YAML paths selects the crisis regime (e.g. `heston_crisis.yaml`, `merton_crisis.yaml`, `liquidity_crisis.yaml`). The composed recipe exposes deterministic seeds, Heston/SABR parameters, optional jump overlays and liquidity spread helpers that downstream simulators consume.

### Validation & Expected Behaviour

Two pytest modules document calibration fidelity and the qualitative jump to crisis dynamics:

- `tests/sim/test_calibration_moments.py` checks annualised moments, variance persistence and jump frequency for calm/crisis with and without Merton jumps. It also writes `runs/sim_tests/sim_manifest.json` with the observed stats for reproducibility.
- `tests/sim/test_pricing_sanity.py` confirms European call prices and implied vols rise from calm to crisis and that SABR crisis parameters widen the smile.

Run the CI-light suite with:

```bash
pytest -m "not heavy" tests/sim
```

The heavy 50k-path moment test is marked `@pytest.mark.heavy` for local benchmarking.

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

## Reproduce the paper

The paper harness automates the full cross-product of methods, seeds, and evaluation windows that back the reported metrics.

- Ensure the packaged SPY slice (`data/spy_sample.csv`) and Hydra configs under `configs/train/` and `configs/eval/` are present. The driver refuses to start if any prerequisite is missing.
- Run the full protocol with:
  ```bash
  make paper
  ```
  Results are written under `runs/paper/` with one directory per method/seed plus nested evaluation windows. The directory also captures a consolidated `final_metrics.json` and provenance manifest `paper_provenance.json` describing the git SHA, environment, and resolved config grid.
- For a command preview without execution, use `make paper DRY=1`.
- For a quick CI-friendly sweep (single seed, smoke configs) run `make paper SMOKE=1`.

## Reproducibility checklist

- Deterministic seeds for training, evaluation, and tests (`seed_list.txt` and Hydra configs).
- Resolved Hydra configs saved under each run directory (`runs/<timestamp>/config.yaml`).
- Metrics logged per-step (`metrics.jsonl`) and in aggregate (`final_metrics.json`) including CVaR-95, Sharpe, and turnover.
- Run metadata captured in `metadata.json` with git commit, platform, Python, and PyTorch versions.
The commands above execute in minutes on a single CPU-only workstation; see the
[reproduction playbook](REPRODUCE.md) for the precise runtime profile and
hardware that were used for the reference paper snapshot.

## Data acquisition summary

The experiments rely on SPY option market snapshots bucketed into volatility
regimes. A 5,000-row smoke subset is included as `data/spy_sample.csv` to enable
local testing and CI. Running `make data` copies this file into
`outputs/paper_data/` so the paper configs can locate it without mutating the
raw download.

For the full paper reproduction you must supply the institutional SPY dataset
cited in the paper (2017–2022 daily close-to-close options). Place the CSV (or
parquet) export in `data/` and update `configs/data/real_spy.yaml` with the
filename if it differs from the default. Provenance expectations are documented
in [REPRODUCE.md](REPRODUCE.md#provenance-and-artifact-tracking).

## Paper pipelines

### `make paper`
Runs `scripts/run_of_record.sh`, which trains the compact IRM head model on the
paper configuration (`configs/train/paper.yaml`) and evaluates the resulting
checkpoint with the matching evaluation profile (`configs/eval/paper.yaml`).
Outputs are written to `runs/paper/` (training) and `runs/paper_eval/`
(evaluation), including `final_metrics.json`, per-environment diagnostics, and
Hydra config mirrors.

### `make report`
Generates the full multi-seed aggregation described in the paper using
`configs/report/default.yaml`. This target expects 30-seed runs under `runs/*`
(as produced by the large-scale sweeps) and renders the complete scorecard,
heatmaps, QQ plots, and optional I–R–E 3D projections into
`outputs/report_assets/`. The command is heavier and is not required for the
single-seed smoke reproduction.

### `make report-paper`
Aggregates the latest paper run into camera-ready assets using
`configs/report/paper.yaml`. The output directory `outputs/report_paper/`
contains the scorecard CSV/LaTeX tables, heatmaps, and provenance manifest for
inclusion in the paper appendix.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Missing data/spy_sample.csv` | Dataset not staged | Download or symlink the SPY snapshot into `data/` and re-run `make data`. |
| `Could not locate the latest training run` | `make paper` was interrupted before finishing | Remove partial directories under `runs/paper/` and re-run `make paper`. |
| `No checkpoints saved in run directory` | Training failed before writing `checkpoints/` | Check `runs/paper/*/metrics.jsonl` for stack traces and re-run after addressing the error. |
| `aggregate.py` exits with "No seed files" | Evaluation artefacts missing | Ensure `make paper` completed successfully and `runs/paper_eval/` contains `diagnostics_seed_*.csv`. |

If an issue persists, capture the failing command output and open a discussion
in the repository.

## FAQ

**Is the repository open source?**  Yes. The code is released under the MIT
License, allowing research and commercial use with attribution. See
[LICENSE](LICENSE) for the exact terms.

**What if I get a `ValueError` about CSV columns when running `make report`?**
This indicates a malformed or truncated diagnostics CSV—usually because the
underlying dataset export was interrupted. Re-run `make paper` (or the
multi-seed sweep) after verifying that your SPY dataset is complete and matches
the schema defined in `configs/data/real_spy.yaml`.

**Can I redistribute the SPY dataset?**  No. The SPY market data is licensed
from a commercial provider and cannot be redistributed. The repository only
ships the 5,000-row smoke subset for testing; you must obtain the full dataset
under your own agreement.

## Additional references

- [REPRODUCE.md](REPRODUCE.md): command-by-command reproduction checklist with
  runtime, hardware, and provenance notes.
- [`experiments/`](experiments/): original research plans, baselines, and phase
  summaries for historical context.
- [`scripts/run_of_record.sh`](scripts/run_of_record.sh): orchestration script
  used by `make paper`.
