# Synthetic Crisis Stress Toggles

This document summarises the configuration knobs introduced for the crisis simulation track.

## Overview

The synthetic generator now composes a Heston stochastic volatility backbone with two optional stress switches:

* **Merton jump overlay** introduces rare but severe log-return shocks that thicken the left tail.
* **Liquidity stress** links spreads and slippage to volatility and trade size, reducing realised PnL during crisis episodes.

Both toggles are deterministic, gated by configuration, and tagged in the output metadata.

## Configuration (`configs/examples/sim_crisis.yaml`)

Key sections:

- `data.heston`: base parameters for the Heston paths (`s0`, `v0`, `kappa`, `theta`, `sigma`, `rho`, `dt_days`, `days`).
- `data.regimes.bands`: defines named volatility buckets used across splits.
- `data.stress.jump`: enables the Merton overlay with intensity (`lam`), mean (`mu_j`), volatility (`sigma_j`), and a list of regimes that receive the stress.
- `data.stress.liquidity`: controls the liquidity cost model with base spread, volatility slope, size slope, and slippage coefficient. `apply_to` restricts the regimes affected.
- `data.episode`: number of episodes per regime and base seed. The seed is combined with deterministic regime offsets so each regime has reproducible randomness.
- `train`: lists which regimes belong to train/val/test splits.

## Outputs

Running `python -m tools.quick_run --config configs/examples/sim_crisis.yaml` creates artefacts under `reports/artifacts/<timestamp>_sim_crisis/` with the structure:

```
reports/artifacts/<timestamp>_sim_crisis/
  seeds/<seed>/<split>/<regime>/
    pnl.csv
    cvar95.json
    sim_params.json
    stress_summary.json
```

Metadata files contain:

- `sim_params.json`: exact generator parameters, stress flags, and episode counts for provenance.
- `stress_summary.json`: realised jump statistics (count, frequency, mean, std) plus liquidity metrics (mean spread in bps, mean slippage cost, turnover).

These artefacts allow downstream diagnostics to associate results with the stress regime via tags (`source=sim`, `regime_name`, `stress_jump`, `stress_liquidity`).

## Determinism

Randomness for Heston, jumps, and liquidity sampling derives from the base seed plus deterministic `adler32` offsets of the regime name. This ensures the same config and seed produce identical outputs across runs.

## Extensibility

Additional regimes or stress parameters can be added by editing the YAML. The registry automatically tags new regimes, and the quick-run utility will emit the necessary provenance files without code changes.
