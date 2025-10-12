# Figures Pipeline

This repository exposes a reproducible post-processing pipeline that consumes the
diagnostic tables produced in Track 4 and emits publication-ready figures. All
commands operate on a single run directory (`runs/<timestamp>_<expname>/`).

## Common CLI Flags

Every plotting script shares the same interface:

```
--run_dir PATH               # required; run root containing tables/
--out_dir PATH               # optional; defaults to <run_dir>/figures
--dpi 300                    # output resolution
--format png,pdf             # comma-separated list of formats
--style journal|poster       # preset Matplotlib styles (default journal)
--seed_filter all|1,2,3      # optional filter across seeds
--regime_filter all|NAME,... # optional regime filter
```

### Available Scripts

| Script | Output name | Source tables | Description |
| --- | --- | --- | --- |
| `plot_invariance_vs_ig.py` | `fig_invariance_vs_ig` | `tables/invariance_diagnostics.csv` | Scatter of invariance (ISI) vs influence gap with regression fit and Pearson correlation. |
| `plot_capital_efficiency_frontier.py` | `fig_capital_efficiency_frontier` | `tables/capital_efficiency_frontier.csv` | Risk–return curves per model (mean PnL vs CVaR-95) with ER annotations. |
| `plot_ire_scatter_3d.py` | `fig_ire_scatter_3d` (+ optional `_byregime`) | `tables/diagnostics_summary.csv` | 3D scatter over I–R–E geometry with turnover-sized markers and regime colours. |
| `plot_regime_panels.py` | `fig_regime_panels` | `tables/diagnostics_summary.csv` | Bar-chart panels comparing models across CVaR-95, IG, ER, and TR for each regime (test split). |
| `plot_alignment_curves.py` | `fig_alignment_curves` | `tables/alignment_head.csv` | Alignment cosine trajectories with penalty curve overlay. |
| `make_all_figures.py` | all of the above | Multiple | Convenience wrapper that regenerates the full figure suite and manifest. |

`plot_ire_scatter_3d.py` provides additional options:

```
--eff_axis ER|composite      # choose efficiency axis (default ER)
--composite_alpha FLOAT      # weight for composite efficiency metric
--separate_by_regime         # emit small multiples suffixed with _byregime
```

## Example Commands

Generate all figures for the latest smoke run:

```bash
RUN_DIR=$(readlink -f runs/latest)
python scripts/make_all_figures.py --run_dir "$RUN_DIR"
```

Regenerate only the capital-efficiency frontier with a poster layout:

```bash
python scripts/plot_capital_efficiency_frontier.py \
  --run_dir runs/20240101_paper \
  --style poster \
  --format pdf
```

Create I–R–E small multiples for selected regimes:

```bash
python scripts/plot_ire_scatter_3d.py \
  --run_dir runs/20240101_paper \
  --regime_filter crisis_a,crisis_b \
  --separate_by_regime
```

All scripts write outputs to `<run_dir>/figures/` (or `--out_dir` if provided) and
update `manifest.json` with the generated file names and input provenance.

