# Reproduction Checklist

The baseline phase-1 experiments (ERM, ERM-reg, IRM, GroupDRO, V-REx) can be re-run using the convenience scripts provided in this
repository. All commands mirror metrics locally under `runs/<timestamp>/` and, when desired, can stream updates to Weights &
Biases (W&B).

## Disable or enable W&B logging

Local mirrors are always written under `runs/<timestamp>/`. To keep runs fully offline pass `logger.use_wandb=false` (this is the
default) to any Hydra-driven command:

```bash
scripts/run_train.sh train/erm logger.use_wandb=false
scripts/run_eval.sh train/erm logger.use_wandb=false eval.report.checkpoint_path=/path/to/checkpoint.pt
```

When you want to send metrics to W&B, flip the flag:

```bash
scripts/run_train.sh train/erm logger.use_wandb=true
```

Both modes produce identical `final_metrics.json`, `metadata.json`, checkpoints, and artefacts inside `runs/<timestamp>/` so you
can compare results or archive them locally regardless of the remote logging state.

## Phase 1 baseline sweep

```bash
scripts/make_reproduce.sh
```

By default this replays the ERM, ERM-reg, IRM, GroupDRO, and V-REx configurations with the recommended deterministic seed. Append
Hydra overrides as needed, for example to disable W&B across the sweep:

```bash
scripts/make_reproduce.sh logger.use_wandb=false
```

The script will train each configuration, locate the freshest checkpoint, and immediately evaluate it on the crisis regime. All
results are mirrored locally under `runs/` and the generated reports land in `outputs/_baseline_erm_base/`.
