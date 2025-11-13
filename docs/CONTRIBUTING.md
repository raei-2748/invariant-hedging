# Contributing

Thanks for your interest in improving the invariant-hedging project! This guide covers the
expected workflow and conventions.

## Development workflow

1. Fork the repository and create a feature branch named `feature/<summary>` or `fix/<summary>`.
2. Install the pinned dependencies:
   ```bash
   python3 -m pip install -r requirements-lock.txt
   python3 -m pip install -e .[dev]
   ```
   Conda users can replicate the environment with:
   ```bash
   conda env create -f environment.yml
   ```
3. Format and lint using Ruff before sending a PR:
   ```bash
   python3 -m ruff check src tests tools experiments
   ```
4. Run the full test matrix locally:
   ```bash
   make tests
   make smoke-check
   make paper SMOKE=1
   make report-paper ARGS="--smoke"
   ```
   The smoke harness compares metrics with numerical tolerances (`atol=1e-7`, `rtol=1e-5`) and should not report any drift.

## Code style

- Python code follows Ruff's defaults plus type annotations where practical.
- Avoid global state; prefer dependency injection via Hydra configs.
- Keep Hydra configs composable and prefer overriding via `train=<variant>`.

## Pull requests

- Reference the GitHub issue when applicable.
- Summarise user-facing changes and include relevant metrics or screenshots.
- CI must pass before requesting review. Branch protection requires the **Lint**
  and **Tests** workflows to succeed on every pull request. The nightly **Pipeline
  Smoke** workflow exercises `make paper SMOKE=1` and `make report-paper` for
  end-to-end regression coverage.
