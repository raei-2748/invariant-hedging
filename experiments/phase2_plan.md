# Phase 2 Plan: Head-Only IRM and Diagnostics

**Goal:**  
Evaluate whether applying the IRM regularization only to the hedge head improves Crisis CVaR-95 relative to ERM and V-REx while maintaining turnover within +20 %.

## 1. Environments
- Train → Low + Medium  
- Validate → High  
- Hold-out → Crisis  

## 2. Models and Sweeps
| Model | Key Params | Config Alias |
|--------|------------|--------------|
| ERM | baseline | `train/erm` |
| IRM-head | λ ∈ {1e-2, 1e-1, 1} | `train/irm_head` |
| V-REx | β ∈ {1, 5, 10} | `train/vrex` |

Example sweep:
```bash
make sweep model=irm_head lambda_grid="[1e-2,1e-1,1]"
```

## 3. Metrics
- **Primary:** CVaR-95 of P&L
- **Secondary:** Mean P&L, Turnover, Sharpe

## 4. Success Criterion
- ≥ 10 % Crisis CVaR-95 improvement vs ERM
- ≤ 20 % increase in turnover

## 5. Diagnostics to Log
- IG (Invariance Gap)
- WG (Worst-Group Variance)
- MSI (Mutual Stability Index)

## 6. Outputs
- Store results in `outputs/_phase2_headIRM/`
- Each run auto-emits `scorecard.json` and `metrics.jsonl`

## 7. Paper Note
- Figure targets: CVaR frontier, QQ plot (ERM vs IRM), λ-sweep
