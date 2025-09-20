# invariant-hedging

Invariant Risk Minimization for Deep Hedging under regime shifts.

## Hedging IRM Research Experiment

This repository contains a simulation and training framework for studying invariant risk minimization (IRM) and related baselines on volatility-regime hedging tasks. The implementation follows the research design laid out in `Research Setup.md`.

## Getting Started

1. Install dependencies:
   ```bash
   make setup
   ```
2. Launch a default IRM training run:
   ```bash
   make train
   ```
   Hydra logs outputs to `outputs/runs/<timestamp>`.
3. Evaluate a saved checkpoint (replace the path with your checkpoint file):
   ```bash
   make evaluate CHECKPOINT=outputs/checkpoints/checkpoint_150000.pt
   ```

## Project Structure

- `configs/`: Hydra configuration hierarchy for data, models, algorithms, and evaluation.
- `src/hirm_experiment/data/`: Synthetic volatility-regime simulator and dataset wrappers.
- `src/hirm_experiment/models/`: Hedging policy networks.
- `src/hirm_experiment/algorithms/`: Implementations of ERM, IRM, GroupDRO, and VREx training objectives.
- `src/hirm_experiment/training/`: Training engine, logging, and learning-rate scheduling utilities.
- `src/hirm_experiment/evaluation/`: Metric computation, evaluator logic, and analysis helpers.
- `src/hirm_experiment/analysis/`: Reporting utilities for coverage tables and spread sensitivity curves.

## Key Outputs

During evaluation the CLI reports:
- Per-environment risk metrics including CVaR-95, mean P&L, turnover, drawdown, and Sharpe ratio.
- Coverage statistics (episodes, horizon, average realized volatility, and transaction cost settings).
- Spread-sensitivity data to support the required robustness plot.

These artifacts can be consumed by downstream notebooks or plotting scripts under `src/hirm_experiment/analysis`.
