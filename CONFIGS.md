# Hydra configuration primer

The training and evaluation entry points load `configs/experiment.yaml`, which composes the
following top-level groups. Key parameters you may want to override:

- `data`: Selects the dataset generator and loader. `synthetic` spawns GBM/Heston regimes.
- `envs`: Lists training/validation/test environments by name. Override to balance regimes.
- `model`: Controls the policy architecture and objective (`erm`, `irm`, `hirm`, `vrex`).
- `train`: Epoch-level hyperparameters such as `steps`, `batch_size`, `grad_clip`, and `seed`.
- `irm`: Additional penalty settings (`lambda_target`, `schedule`, `warmup_steps`) when IRM-style
  penalties are active.
- `loss`: Risk objective parameters (e.g. `cvar_alpha`).
- `logging`: Controls metric frequency and local mirror directory.
- `eval`: Evaluation environments and report options.

Override any of these from the command line:

```bash
python scripts/train.py train=phase2 irm.lambda_target=0.1 train.seed=7
```

Hydra resolves the config and writes the final structure to `runs/<timestamp>/config.yaml` for
post-hoc analysis.
