# Phase 2 — Head-Only IRM + Diagnostics

Phase 2 extends the invariant hedging study beyond the ERM baselines frozen in Phase 1. The focus is on stress-testing Invariant Risk Minimisation (IRM) and V-REx objectives when the representation is frozen and only the policy head adapts to regime shifts.

## Objectives

- **Head-only IRM sweeps:** explore \(\lambda \in \{10^{-2}, 10^{-1}, 1\}\) while keeping the feature extractor fixed to the ERM baseline.
- **V-REx diagnostics:** sweep \(\beta \in \{1, 5, 10\}\) and log variance penalties alongside tail metrics.
- **Stress environments:** introduce jump intensity shocks combined with liquidity squeezes to expose failure modes not covered in Phase 1.
- **Early diagnostics:** capture Integrated Gradients (IG), Wasserstein Gradients (WG), and Market Stability Indicators (MSI) throughout training to understand representation drift.

## Milestones

1. **Reproduction harness** — integrate Phase-2 configs into the smoke/sanity tooling so they can be exercised in CI and on workstations.
2. **Head-only IRM baseline** — finalise data loaders and schedulers for the head-only constraint and release tuned checkpoints.
3. **Diagnostic logging** — wire IG/WG/MSI hooks into the evaluation runner with reproducible seeds.
4. **Benchmark release** — publish crisis CVaR-95, Sharpe, turnover, and diagnostic summaries under `outputs/_phase2_snapshot/`.

## Reproducibility notes

- All experiments should use the pinned `requirements.txt` / `environment.yml` shipped with the repository.
- Metadata (`metadata.json`) must include git SHA, torch version, and environment fingerprints for every Phase-2 run.
- Keep seeds frozen at 0 for smoke checks; broader sweeps can expand the seed list after verification.

For status updates and detailed experiment tracking, synchronise notes with the main project board and surface blockers via issues tagged `phase2`.
