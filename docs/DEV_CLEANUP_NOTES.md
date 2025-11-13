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

### Directories removed
- Removed the monolithic `src/invariant_hedging/core/`, `modules/`, and `legacy/` trees; all living code now resides under `src/invariant_hedging/{cli,data,diagnostics,evaluation,reporting,runtime,training,visualization}`.
- Dropped the bespoke `src/invariant_hedging/hirm/` shim after folding its seeding helpers into `runtime/seed.py`.

### Modules moved
- Data ingestion and simulators formerly under `modules/data/*`, `modules/sim/*`, and `legacy/envs/*` now live in `invariant_hedging.data` (including `data/envs`, `data/real`, `data/sim`, and `data/markets`).
- Diagnostics and reporting helpers were consolidated from `legacy/diagnostics*`, `evaluation/analyze_diagnostics.py`, and `legacy/report_*` into `diagnostics/`, `reporting/core`, and `reporting/legacy`.
- Training/optimization code from `core/engine.py`, `core/losses.py`, `core/optimizers.py`, `modules/models.py`, and `legacy/objectives/*` migrated into `training/engine.py`, `training/losses.py`, `training/optimizers.py`, `training/models/`, and `training/objectives/`.
- Runtime utilities (device resolution, checkpointing, logging, stats) now live in the dedicated `runtime/` package and are re-exported via `invariant_hedging.runtime`.
- CLI entry points for training/evaluation/report generation were centralized under `src/invariant_hedging/cli/` for Hydra to target.

### Modules deleted
- Removed dead legacy stacks including `tools/quick_run.py`, `tools/scripts/compute_diagnostics.py`, `tests/test_hirm_head.py`, and the entire `legacy/diagnostics_v2` + `legacy/train` hierarchy.
- Deleted stale `src/invariant_hedging/core/*` readers/writers and redundant `modules/README.md`/`legacy/README.md` placeholders per the destructive refactor plan.

### Config updates and archival
- Archived every unused or broken Hydra preset under `archive/configs/`, covering `configs/algorithm`, `configs/examples`, `configs/evaluation`, `configs/train/examples`, `configs/training`, `configs/run`, and all single-file presets marked “Unused/Broken” in CODE_AUDIT (e.g., `configs/data/real_anchor.yaml`, `configs/envs/high_eval.yaml`, `configs/eval/high.yaml`, `configs/model/delta*.yaml`, `configs/train/phase2*.yaml`, `configs/logging/wandb.yaml`, `configs/reproduce.yaml`, `configs/experiment_eval.yaml`).
- Retained only the production Hydra trees consumed by the new CLI entry points (e.g., `configs/base.yaml`, `configs/train/{erm,irm,...}.yaml`, `configs/eval/{daily,robustness,smoke}.yaml`).

### Tests updated or removed
- Updated the entire pytest suite to import from the new `invariant_hedging.{data,diagnostics,evaluation,reporting,runtime,training}` namespaces.
- Dropped `tests/test_hirm_head.py`, which referenced the removed `hirm_head` objective, and refreshed fixtures under `tests/data/report/` to cover the new report schema.

### Merged utilities and structural notes
- Unified deterministic seeding helpers by merging `invariant_hedging.hirm.utils.determinism` into `runtime/seed.py`, exposing `resolve_seed`, `set_seed`, and generator helpers via `invariant_hedging.runtime`.
- Added `runtime/__init__.py` exports so downstream code can rely on `invariant_hedging.runtime.seed` rather than duplicating import shims.
- Adjusted `tools/run_of_record.sh`, `tools/run_eval.sh`, `tools/run_train.sh`, and the Makefile to export `PYTHONPATH=src` so Hydra entry points work without editable installs.
- Calibrated `tools/report/generate_report.py` smoke-mode logging to degrade warnings when diagnostics or `final_metrics.json` are intentionally absent.
- No divergence from the planned hierarchy was required; all surviving modules fit within the canonical `src/invariant_hedging/` package tree described in Phase 1.
