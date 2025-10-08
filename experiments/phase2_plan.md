# Phase-2 Experiment Plan

Phase-2 introduces fully reproducible grids for the invariant hedging study. All runs must use
`requirements.lock.txt` and the Docker image included in the repository to guarantee identical
software stacks across machines.

## Experiment matrix

| Method alias | Hydra entry point          | Critical grid                                                   |
|--------------|----------------------------|-----------------------------------------------------------------|
| `erm_reg`    | `python -m src.train`      | `model.objective=erm_reg`, `train.lambda=[0.0]`                 |
| `irm_head`   | `python -m src.train`      | `model.objective=irm_head`, `irm.lambda=[0.01,0.1,1.0,10.0]`    |
| `groupdro`   | `python -m src.train`      | `model.objective=groupdro`, `groupdro.step_size=[0.01,0.1]`    |
| `vrex`       | `python -m src.train`      | `model.objective=vrex`, `model.vrex.penalty_weight=[1,5,10]`   |

Launch each command with `python -m src.train method=<alias> seed=<seed> +phase=phase2`.

Seeds are sourced from `seeds/seed_list.txt` (currently `1, 2, 3, 4, 5`). When launching a grid,
run every configuration for all seeds and log outputs under `runs/phase2/`.

## Selection rule

Tune on the High-vol validation environment. For each method, pick the hyper-parameter setting
with the best High-vol CVaR-95 (lower is better). Report diagnostics using that selection and
propagate the chosen configuration to the Crisis hold-out evaluation.

## Required diagnostics

Each run must record:
- CVaR-95, mean PnL, Sortino ratio, turnover
- Invariance Gap (IG), Worst-Group Gap (WG), Mechanism Sensitivity Index (MSI)
- Git commit hash and environment fingerprint (Torch + CUDA versions)

Use `python -m src.diagnostics.collect --runs runs --out tables/diag.csv` followed by
`python -m src.diagnostics.plot --csv tables/diag.csv` to consolidate results.

## Reproduction command

Once the repository is prepared, Phase-2 can be reproduced in one command:

```bash
make phase2
```

This executes every method/seed combination, evaluates the Phase-2 split, aggregates diagnostics,
and generates plots under `figures/`.
