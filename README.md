# HIRM

[![CI](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml/badge.svg)](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml)
[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10--3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper reproduction guide](https://img.shields.io/badge/reproduce-paper-blue.svg)](docs/REPRODUCE.md)

HIRM (Hedging with IRM) is a research codebase for reproducing the paper results on
robust hedging under regime shifts. This repository now defaults to the
**paper reproduction workflow**: acquiring the SPY dataset snapshot, running the
compact training/evaluation pipeline, and generating the camera-ready tables and
figures.

For a full walkthrough of every command, expected artefacts, and provenance
requirements see [docs/REPRODUCE.md](docs/REPRODUCE.md).
Supporting guides live under [docs/](docs/README.md).


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
   tools/run_of_record.sh --dry-run
   ```
4. **Train + evaluate the paper configuration on CPU**
   ```bash
   make paper
   ```
5. **Build publication tables/figures from the paper run**
   ```bash
   make report-paper
   ```
6. *(Optional)* **Rebuild the full report assets for multi-seed runs**
   ```bash
   make report
   ```

## Paper-aligned layout

```
invariant-hedging/
├── src/
│   ├── core/              # Optimisation engine and reusable losses
│   ├── modules/           # Invariance, robustness, and data modules
│   ├── evaluation/        # Crisis evaluation and diagnostics entrypoints
│   ├── visualization/     # Plotting helpers for the manuscript figures
│   └── legacy/            # Archived papers' scripts retained for provenance
├── experiments/           # Thin CLIs that launch training/diagnostics runs
├── configs/               # Hydra configuration tree (defaults, sweeps, methods)
├── reports/               # Generated tables/figures/logs from evaluations
└── meta/                  # Provenance metadata and archived pre-release bundles
```

| Paper concept | Implementation |
| --- | --- |
| Invariance (gradient alignment heads) | `src/core/losses.py`, `src/modules/head_invariance.py` |
| Robustness diagnostics | `src/modules/diagnostics.py`, `src/evaluation/analyze_diagnostics.py` |
| Efficiency frontiers & crisis evaluation | `src/evaluation/evaluate_crisis.py`, `src/visualization/` |
| Training engine & orchestration | `src/core/engine.py`, `experiments/run_*.py` |

Legacy utilities (including the original `train/` loop and report builders) now live under `src/legacy/` and remain importable for provenance-sensitive experiments.
### Figures

Regenerate the Phase-2 diagnostic figures from either a per-seed scoreboard or the aggregated `scorecard.csv` produced by `make report`:

```bash
RUN_DIR=$(readlink -f runs/latest)
python src/visualization/plot_cvar_by_method.py --run_dir "$RUN_DIR" --out_dir reports/figs
python src/visualization/plot_diag_correlations.py --run_dir "$RUN_DIR" --out_dir reports/figs
python src/visualization/plot_capital_frontier.py --run_dir "$RUN_DIR" --out_dir reports/figs
```

Outputs are written to `reports/figs/` (or `--out_dir`) alongside `.meta.json` files capturing the exact filters and inputs.

2. Provide diagnostic batches (held-out from training) via
   `diagnostics.probe`. Batches are dictionaries containing `risk`, `outcome`,
   `positions`, and optionally `grad`/`representation` tensors per environment.

3. Run evaluation. The export helper will assemble I–R–E metrics using the
   modules in `src/modules/diagnostics.py` and log the output CSV path via the existing
   run logger.

## Smoke test

The helper scripts export conservative Intel OpenMP settings so PyTorch runs even in sandboxes with no shared-memory segment. Kick off the short training loop with:

```bash
tools/run_train.sh train/smoke
```

If you prefer to invoke Python directly, mirror those defaults explicitly:

```bash
OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE python3 experiments/run_train.py --config-name=train/smoke
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

`tools/make_reproduce.sh` re-runs the ERM, ERM-reg, IRM, GroupDRO and V-REx configurations for seed 0, evaluates the best checkpoint for each on the crisis environment, and regenerates the crisis CVaR-95 table plus QQ plots. All seeds are controlled via `configs/train/*.yaml` and `src/core/utils/seed.py` to guarantee deterministic `metrics.jsonl` for `seed=0`.

## Reproduce the paper

The paper harness automates the full cross-product of methods, seeds, and evaluation windows that back the reported metrics.

- Ensure the packaged SPY slice (`data/spy_sample.csv`) and Hydra configs under `configs/train/` and `configs/eval/` are present. The driver refuses to start if any prerequisite is missing.
- Run the full protocol with:
  ```bash
  make paper
  ```
  Results are written under `runs/paper/` with one directory per method/seed plus nested evaluation windows. The directory also captures a consolidated `final_metrics.json` and provenance manifest `meta/paper_provenance.json` describing the git SHA, environment, and resolved config grid.
- For a command preview without execution, use `make paper DRY=1`.
- For a quick CI-friendly sweep (single seed, smoke configs) run `make paper SMOKE=1`.

## Reproducibility checklist

- Deterministic seeds for training, evaluation, and tests (`seed_list.txt` and Hydra configs).
- Resolved Hydra configs saved under each run directory (`runs/<timestamp>/config.yaml`).
- Metrics logged per-step (`metrics.jsonl`) and in aggregate (`final_metrics.json`) including CVaR-95, Sharpe, and turnover.
- Run metadata captured in `metadata.json` with git commit, platform, Python, and PyTorch versions.
The commands above execute in minutes on a single CPU-only workstation; see the
[reproduction playbook](docs/REPRODUCE.md) for the precise runtime profile and
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
in [docs/REPRODUCE.md](docs/REPRODUCE.md#provenance-and-artifact-tracking).

## Paper pipelines

### `make paper`
Runs `tools/run_of_record.sh`, which trains the compact IRM head model on the
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

- [docs/REPRODUCE.md](docs/REPRODUCE.md): command-by-command reproduction checklist with
  runtime, hardware, and provenance notes.
- [`experiments/`](experiments/): original research plans, baselines, and phase
  summaries for historical context.
- [`tools/run_of_record.sh`](tools/run_of_record.sh): orchestration script
  used by `make paper`.
