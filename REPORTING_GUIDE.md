# Reporting extension guide

This document explains how to extend the reporting stack while keeping
plots, tables, and schemas compatible with the existing paper release.
It assumes you already ran the aggregation pipeline via
`scripts/aggregate.py` or `make report-paper`.

## Directory tour

| Component | Purpose |
| --- | --- |
| `configs/report/*.yaml` | Declarative description of which seeds, metrics, and figures to render. |
| `scripts/aggregate.py` | CLI entrypoint that loads a report config and writes aggregates to `outputs/`. |
| `src/report/aggregate.py` | Core data loader that merges diagnostics CSVs and `final_metrics.json` payloads. |
| `src/report/plots.py` | Matplotlib helpers for scorecards, heatmaps, QQ plots, and seed distributions. |
| `src/report/schema.py` | Validation helpers that enforce a stable `final_metrics.json` schema. |

## Adding new plots or modifying visuals

1. **Decide where the plot belongs**. Heatmap/table tweaks usually live in
   `plot_heatmaps` or `plot_scorecard`. New figure families can be added
   as additional functions in `src/report/plots.py`.
2. **Update the report config** to drive the plot. For example, extending
   the invariance block in [`configs/report/paper.yaml`](configs/report/paper.yaml)
   automatically forwards the new metric list to `plot_heatmaps` and the
   scorecard renderer.
3. **Respect the existing style helpers**. All plotting functions call
   `_apply_style`, so new code should reuse it for consistent fonts,
   colours, and DPI (`src/report/plots.py`).
4. **Write the artefacts to the configured directory**. Use the provided
   `_ensure_dir` helper so figures land in the same folder as existing
   outputs (typically `outputs/report_paper/figures/`).
5. **Add tests where possible**. Lightweight plotting tests can assert
   that functions execute without raising and produce expected filenames.

## Keeping `final_metrics.json` stable

The report pipeline trusts every run directory to expose a
`final_metrics.json` that conforms to `src/report/schema.py`.

- When you introduce a new metric, **use `validate_final_metrics`** (or
  `load_final_metrics`) to normalise the payload before writing it to
  disk. This guarantees numeric values, finite floats, and a valid
  `schema_version` field.
- **Bump `SCHEMA_VERSION`** only when making a breaking change. Increment
  the patch version for additive metrics, the minor version for backwards
  compatible structural additions, and the major version for
  incompatible rewrites.
- **Record metadata** inside the payload instead of inventing new file
  formats. `validate_final_metrics` separates scalar metrics from nested
  metadata so downstream tooling can survive schema evolution.

## Extending aggregation logic

`src/report/aggregate.py` drives the heavy lifting:

1. `resolve_seed_directories` collects run directories that contain
   `diagnostics_manifest.json`.
2. `select_seeds` deterministically subsamples seeds when the config
   limits how many to aggregate.
3. `load_seed_dataframe` combines diagnostics CSVs and the validated
   `final_metrics.json` into a tidy DataFrame.

To add a new aggregation step:

- Extend `load_seed_dataframe` to include additional metadata columns or
  to ingest new diagnostics files. Keep the output tidy (columns: regime,
  metric, value, seed, …) so existing plots keep working.
- Update `ensure_metrics_exist` if you introduce new required metrics so
  misconfigured runs fail fast.
- Return the extra columns in the `AggregateResult` dataclass if the data
  should feed the visualisation layer.

## Recommended development loop

1. Run `make paper SMOKE=1` to produce a tiny dataset of diagnostics.
2. Modify configs/code as described above.
3. Regenerate aggregates with
   `PYTHONPATH=. python scripts/aggregate.py --config configs/report/paper.yaml --lite`.
4. Inspect the updated artefacts under `outputs/` and commit both the
   code and regenerated figures (if tracked).

Following these steps keeps the reporting surface adaptable while
preserving schema guarantees for downstream consumers.
