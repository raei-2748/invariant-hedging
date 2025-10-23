# Contributing

Thanks for your interest in improving the invariant-hedging project! This guide covers the
expected workflow and conventions.

## Development workflow

1. Fork the repository and create a feature branch named `feature/<summary>` or `fix/<summary>`.
2. Install the pinned dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
   Conda users can replicate the environment with:
   ```bash
   conda env create -f environment.yml
   ```
3. Format and lint using Ruff before sending a PR:
   ```bash
   python3 -m ruff check src tests
   ```
4. Run the full test matrix locally:
   ```bash
   make tests
   make smoke
   python tools/scripts/train.py config=train/smoke steps=100 seed=0
   python tools/scripts/train.py config=train/smoke steps=100 seed=0
   python tools/scripts/diff_metrics.py reports/artifacts/latest_0/metrics.jsonl reports/artifacts/latest_1/metrics.jsonl
   ```
   The diff step should report a mean absolute difference below `1e-6`.

## Code style

- Python code follows Ruff's defaults plus type annotations where practical.
- Avoid global state; prefer dependency injection via Hydra configs.
- Keep Hydra configs composable and prefer overriding via `train=<variant>`.

## Pull requests

- Reference the GitHub issue when applicable.
- Summarise user-facing changes and include relevant metrics or screenshots.
- CI must pass before requesting review. Branch protection requires the following
  workflows to succeed on every pull request:
  - **CI Smoke** — linting, unit tests, paper-config smoke train/eval, and the SPY
    data-loader checks.
  - **CI Dependencies** — captures the package environment, CUDA metadata, and the
    `paper_provenance.py` manifest used for reproducibility verification.
