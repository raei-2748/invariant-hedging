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
