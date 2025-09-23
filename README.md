# invariant-hedging

**Research question:** Does adding Invariant Risk Minimization (IRM) to a deep hedging framework improve robustness against regime shifts (such as crisis periods like Volmageddon), compared to a standard deep hedger without IRM?

## Quick start

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
- `metadata.json`: git commit hash and Python version

If W&B credentials are available the same metrics are mirrored to the `invariant-hedging` project; offline mode is supported via `WANDB_MODE=offline`.

## Testing

Unit tests cover pricing Greeks, CVaR estimation, cost kernels and deterministic seeding. Run them with:

```bash
make tests
```

GitHub Actions executes the test suite together with a smoke train/eval pass on every push.

## Data

The synthetic generator supports GBM and Heston dynamics with environment-specific transaction costs. A tiny SPY options slice (`data/spy_sample.csv`) is bundled as a deterministic real-data anchor that exercises the full feature pipeline.

Episode configuration, cost files and model settings live under `configs/`. Adjust these as needed for experiments or sweeps. The default training protocol performs 20k ERM pre-training steps, a 10k IRM ramp, and continues until 150k total updates with environment-balanced batching.

## Reproducibility

`scripts/make_reproduce.sh` re-runs the ERM, ERM-reg, IRM, GroupDRO and V-REx configurations for seed 0, evaluates the best checkpoint for each on the crisis environment, and regenerates the crisis CVaR-95 table plus QQ plots. All seeds are controlled via `configs/train/*.yaml` and `src/utils/seed.py` to guarantee deterministic `metrics.jsonl` for `seed=0`.
