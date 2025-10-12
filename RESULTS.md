# Results Provenance Index

This index links every published figure and table to the script that produced it,
where the artefact lives, and how to retrieve the originating commit SHA. Replace
`<RUN_DIR>` with the absolute path to the run directory (for example,
`runs/20250218_123456_train_paper`). The commit SHA is recorded automatically in
`<RUN_DIR>/run_provenance.json` under `git.commit`.

## Figures

| Figure/Table | Script entry point | Artefact path (relative to `<RUN_DIR>`) | Commit SHA source |
| --- | --- | --- | --- |
| Figure 1 — Penalty sweep | `scripts/report/generate_report.py` → `report.figures.plot_penalty_sweep` | `figures/fig_penalty_sweep.png` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Figure 2 — Head vs feature ablation | `scripts/report/generate_report.py` → `report.figures.plot_head_vs_feature_ablation` | `figures/fig_head_vs_feature.png` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Figure 3 — ISI decomposition | `scripts/report/generate_report.py` → `report.figures.plot_isi_decomposition` | `figures/fig_isi_decomposition.png` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Figure 4 — Cross-regime heatmap | `scripts/report/generate_report.py` → `report.figures.plot_cross_regime_heatmap` | `figures/fig_cross_regime_heatmap.png` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Figure 5 — Capital efficiency frontier | `scripts/report/generate_report.py` → `report.figures.plot_capital_frontier` | `figures/fig_capital_frontier.png` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Figure 6 — CVaR violin | `scripts/report/generate_report.py` → `report.figures.plot_cvar_violin` | `figures/fig_cvar_violin.png` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Figure 7 — IG vs CVaR scatter | `scripts/report/generate_report.py` → `report.figures.plot_ig_vs_cvar` | `figures/fig_ig_vs_cvar.png` | `<RUN_DIR>/run_provenance.json` → `git.commit` |

## Tables

| Figure/Table | Script entry point | Artefact path (relative to `<RUN_DIR>`) | Commit SHA source |
| --- | --- | --- | --- |
| Table 1 — Main scorecard | `scripts/report/generate_report.py` → `report.tables.build_table` (scorecard block) | `tables/scorecard.csv` & `tables/scorecard.tex` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Table 2 — Invariance metrics | `scripts/report/generate_report.py` → `report.tables.build_table` (invariance block) | `tables/invariance.csv` & `tables/invariance.tex` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Table 3 — Robustness metrics | `scripts/report/generate_report.py` → `report.tables.build_table` (robustness block) | `tables/robustness.csv` & `tables/robustness.tex` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Table 4 — Efficiency metrics | `scripts/report/generate_report.py` → `report.tables.build_table` (efficiency block) | `tables/efficiency.csv` & `tables/efficiency.tex` | `<RUN_DIR>/run_provenance.json` → `git.commit` |
| Seed-wise diagnostics | `scripts/report/generate_report.py` → `_generate_tables` | `tables/seed_values.csv` | `<RUN_DIR>/run_provenance.json` → `git.commit` |

### How to answer “Which commit produced Figure 3?”

1. Identify the run directory that contains the published artefacts (e.g. from the
   paper manifest or release notes).
2. Consult the table above: Figure 3 corresponds to `figures/fig_isi_decomposition.png`.
3. Open `<RUN_DIR>/run_provenance.json` and read the `git.commit` field. This is the exact commit that produced the figure.

Because `run_provenance.json` is written for every run, this lookup takes only a
few seconds once the relevant run directory is known.
