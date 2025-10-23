# Cleanup manifest (2025-10-23)

- Created `pre-cleanup-20251023` tag before restructuring.
- Consolidated documentation under `docs/` with a new `docs/README.md` index; updated root README links.
- Archived exploratory notebooks in `legacy/notebooks/` and moved former `config/*.yaml` examples into `configs/examples/`.
- Migrated CLI utilities from `scripts/` to `tools/` and plotting modules into `src/visualization/`; updated imports to use the `src.modules` namespace.
- Removed tracked runtime artefacts (`outputs/`) and added them to `.gitignore`.
- Folded the standalone `diagnostics/` package into `src/modules/diagnostics.py`; retained the previous implementations under `legacy/diagnostics_v2/` for provenance.
- Updated CI to set `PYTHONPATH` explicitly and execute `pytest -m "not heavy"` plus `make paper SMOKE=1`.
- Documented the new layout in the README and ensured helper scripts reference the relocated tools.
