# DEV_CLEANUP_NOTES

## Baseline Failures

- `python -m pip install -r requirements-lock.txt`
  - Fails immediately because the lock file pins `certifi==2025.10.5`, a future release that is unavailable on PyPI in this environment and also blocked by the proxy, so pip cannot satisfy dependencies. 【326c61†L1-L4】
- `pip install -e .[dev]`
  - Cannot finish building dependencies because pip is unable to download the required `setuptools>=67` wheel via the proxy; the editable install aborts during the build dependency bootstrap step. 【f0915e†L1-L27】
- `ruff check --no-fix`
  - Static analysis reports 19 violations, including undefined names, unused imports, and outright syntax errors in `tools/quick_run.py` and `tools/scripts/compute_diagnostics.py`; no fixes were attempted. 【2439f3†L1-L97】
- `pytest -m "not heavy" --maxfail=1 --disable-warnings`
  - Test collection fails immediately: `tests/data/test_real_anchors.py` triggers a circular import between `invariant_hedging.core.utils` and `invariant_hedging.modules.data.real.loader`, preventing the test module from importing. 【7d5c0a†L1-L23】
- `make paper SMOKE=1`
  - The first training job crashes because Python cannot import the `invariant_hedging` package—the editable install never succeeded, so the package is not available on the interpreter path. 【914768†L1-L5】
- `make report-paper`
  - Fails for the same reason as `make paper`: `tools/report/generate_report.py` imports `invariant_hedging`, which is not installed. 【223e6d†L1-L5】

## PHASE 2–3 Refactor Actions
- Removed the obsolete `src/invariant_hedging/core`, `modules`, `legacy`, and `tools/report` trees after relocating all live code into the new `cli/`, `data/`, `training/`, `evaluation/`, `diagnostics/`, `reporting/`, `runtime/`, and `visualization/` packages.
- Promoted the training engine to `invariant_hedging.training.engine`, the crisis evaluation harness to `invariant_hedging.evaluation.crisis`, and the report generator to `invariant_hedging.reporting.cli`; added `invariant_hedging.cli.{train,eval,report}` as the canonical Hydra entrypoints.
- Consolidated runtime helpers (`device`, `checkpoints`, `logging`, `seed`, `stats`, `configs`, `paths`, `io`) under `invariant_hedging.runtime` and rewired training/evaluation imports to consume them directly.
- Moved the data loaders, feature engineering, environment registry, markets, and simulation utilities under `invariant_hedging.data.*` to remove the legacy indirection that previously lived in `modules/`.
- Re-homed all diagnostics, objectives, and IRM/HIRM helpers inside `invariant_hedging.diagnostics.*` and `invariant_hedging.training.{losses,objectives,irm,head_invariance,architectures}` so that no runtime modules import from the deleted legacy package.
- Collapsed the reporting stack by moving `report_core` and `report_assets` into `invariant_hedging.reporting.*` and updating the CLI, aggregation, provenance, and plotting scripts to import from the new namespace.
- Deleted unused config trees (`configs/algorithm`, `evaluation`, `examples`, `run`, `training`), broken presets (`configs/train/phase2*`, `configs/model/hirm_head.yaml`, `configs/logging/wandb.yaml`, etc.), and the unused Hydra `reproduce` entrypoints.
- Removed dead scripts (`tools/quick_run.py`, `tools/scripts/compute_diagnostics.py`, `make_scorecard.py`, `diff_metrics.py`, `prepare_data.py`, `run_baseline.py`) together with tests that referenced the deleted legacy runtime (`tests/test_hirm_head.py`, `tests/test_erm_base_regression.py`, `tests/test_report_schema.py`).
- Updated the Makefile and `experiments/run_{train,diagnostics}.py` to drive the new CLI modules, and repointed every Hydra config `_target_` consumer to the `invariant_hedging` package paths.
- Deleted previously tracked report artifacts under `reports/artifacts/`, replaced the directory with a `.gitkeep`, and added ignore rules so regenerated paper runs remain untracked.
- Rebuilt `tools/run_of_record.sh` to export `PYTHONPATH`, fix the MPS autodetection heredoc, and ensure the inline CSV writer runs without heredoc warnings.
