# Paper-aligned Hydra configs

> **Do not edit without opening a PR.** These YAMLs document the paper's reproducible defaults and must stay version-controlled.

- `data.yaml` — Mirrors the SPY train/validation/crisis windows described in the paper's data splits section, using the same date ranges as [`configs/splits/`](../splits). This records the train span (2017–2019), Q4 2018 validation spike, and the 2018/2020/2022 crisis probes together with the optional 2008 GFC anchor.
- `methods.yaml` — Captures the optimisation protocol from the experimental setup: 30 deterministic seeds, AdamW with cosine warmup, the 150k-step schedule (20k ERM pretrain + 10k IRM ramp), and method-specific knobs such as GroupDRO's step size and V-REx penalties.
- `train.yaml` — Entry point for `python -m src.train --config-name=paper/train`, wiring the paper metadata above into the default Phase 1 ERM training recipe. Override `model=<objective>` to reproduce the other baselines while retaining the documented seeds and horizons.
- `eval.yaml` — Entry point for `python -m src.eval --config-name=paper/eval`, pairing the same seeds with the crisis evaluation windows and horizons (20/60/120 days) cited in the evaluation section. Supply `eval.report.checkpoint_path=<checkpoint>` at runtime to target a trained model.

Both entry points record pointers to the metadata under `paper.*` in the resolved config so downstream tooling or notebooks can locate the canonical splits, seeds, optimisers, and evaluation windows from a single source of truth.
