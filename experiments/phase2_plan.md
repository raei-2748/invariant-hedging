# Phase-2 Experiment Plan

This document locks the Phase-2 sweep space used by `scripts/reproduce_phase2.sh`. All commands assume the
Phase-2 environment from `requirements.lock.txt` and `Dockerfile`.

## Environments
- **Train:** `low`, `medium`
- **Validation:** `high`
- **Test:** `crisis`

## Seed policy
- Deterministic seed list stored in [`seeds/seed_list.txt`](../seeds/seed_list.txt).
- Reproduction scripts iterate the full list for every method.

## Model grids
| Method     | Hydra target                     | Grid                                                                                     |
|------------|----------------------------------|------------------------------------------------------------------------------------------|
| `erm_reg`  | `src.train method=erm_reg`       | `model.regularization.weight_decay ∈ {0.0, 1e-4, 1e-3}`                                   |
| `irm_head` | `src.train method=irm_head`      | `irm.lambda ∈ {1e-2, 3e-2, 1e-1, 3e-1}` with cosine warm-up of 5k steps                  |
| `groupdro` | `src.train method=groupdro`      | `groupdro.step_size ∈ {0.01, 0.05}`, `groupdro.alpha ∈ {0.0, 0.2}`                       |
| `vrex`     | `src.train method=vrex`          | `vrex.beta ∈ {1.0, 5.0, 10.0}`                                                           |

Each configuration inherits the `+phase=2` overrides to align data ranges and logging.

## Selection rule
For each method, select the hyper-parameter setting that minimises **High-vol CVaR-95** on the validation
split. Report the associated crisis metrics and diagnostics for the chosen configuration. All other
artifacts remain available under `runs/` for auditing.

## Reporting
- Persist raw metrics in `runs/<timestamp>/final_metrics.json`.
- Aggregate diagnostics via `python -m src.diagnostics.collect --runs runs --out tables/diag.csv`.
- Generate figures with `python -m src.diagnostics.plot --csv tables/diag.csv`.

## Reproduction checklist
1. Install locked dependencies (`pip install -r requirements.lock.txt`).
2. Run `make phase2` to execute the full sweep over all seeds.
3. Confirm outputs:
   - `tables/diag.csv`
   - `figures/ig_vs_cvar.png`
   - `figures/capital_efficiency.png`
4. Archive the git commit and diagnostics in experiment tracking.
