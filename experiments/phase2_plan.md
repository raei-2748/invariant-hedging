# Phase 2 — Head-Only IRM + Diagnostics

Phase 2 revisits the invariant hedging benchmarks with locked seeds, reproducible grids, and lightweight diagnostics. All runs share the data splits from Phase 1 (train: Low + Medium, validation: High-vol, test: Crisis) and inherit common logging/metadata requirements.

## Experiment matrix

| Method      | Config alias     | Primary sweep                             | Notes |
|-------------|------------------|-------------------------------------------|-------|
| ERM (baseline) | `train/erm_reg`   | none                                      | Regularised ERM control. |
| IRM-Head    | `train/irm_head`  | \(\lambda \in \{10^{-2}, 10^{-1}, 1\}\) | Freeze representation, optimise head with IRM penalty. |
| GroupDRO    | `train/groupdro`  | step size \(\eta \in \{0.01, 0.05\}\)   | Use same horizon/optimizer as ERM. |
| V-REx       | `train/vrex`      | \(\beta \in \{1, 5, 10\}\)              | Variance risk extrapolation baseline. |

**Seed set:** `seeds/seed_list.txt` (1, 2, 3, 4, 5).

## Selection rule

1. Run each method × hyperparameter pair across the seed list.
2. For IRM-Head, select \(\lambda\) using **High-vol CVaR-95**: pick the value with the lowest validation CVaR-95 averaged over seeds.
3. For GroupDRO and V-REx use the same rule (select by High-vol CVaR-95), keeping turnover within +20% of ERM.
4. Report crisis CVaR-95, mean PnL, turnover, IG, WG, and MSI for the chosen hyperparameters.

## Diagnostics

- **IG (Invariance Gap):** spread of train-regime CVaR-95.
- **WG (Worst-Group Gap):** difference between worst test CVaR-95 and worst train CVaR-95.
- **MSI (Mechanism Sensitivity Index):** ratio of sensitivity of invariant head to risk head.

Outputs aggregate to `tables/diag.csv` and plots in `figures/` via `python -m src.diagnostics.collect` and `python -m src.diagnostics.plot`.

## Checklist

- Dependencies resolved through `requirements.lock.txt` or the Docker image.
- All runs store metadata with git SHA and torch version.
- CI smoke suites exercise `make smoke-train`, `make smoke-eval`, and `make phase2`.
