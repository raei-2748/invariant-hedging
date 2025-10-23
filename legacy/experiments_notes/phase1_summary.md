# Phase 1 Summary: ERM Baseline

**Scope:** Establish the ERM-v1 baseline for hedging under the Phase 1 protocol.

## Key Runs
- Default ERM training via `make train CONFIG=train/erm`
- Reproduction sweep `tools/make_reproduce.sh` (ERM, ERM-reg, IRM, GroupDRO, V-REx) stored under `outputs/_baseline_erm_base/`

## Takeaways
- Crisis CVaR-95 establishes the benchmark for Phase 2 comparisons.
- Turnover and Sharpe tracked in `reports/data_ingest_report.md` and `reports/runlog.md` for reproducibility.
- Hydra configs anchored at `configs/train/erm.yaml` and evaluation at `configs/eval/daily.yaml`.

## Next Steps
- Transition to Phase 2 experiments per `legacy/experiments_notes/phase2_plan.md`.
- Keep Phase 1 artifacts untouched as regression anchor for future phases.
