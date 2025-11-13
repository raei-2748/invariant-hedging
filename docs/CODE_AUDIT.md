# CODE_AUDIT

## 1.1 FILE-LEVEL ANALYSIS

### File-level dead code
- **Visualization-only modules.** A dozen CLIs under `src/invariant_hedging/visualization/` (for example `plot_cvar_by_method.py`) are never imported outside their own package exports; `rg` finds matches only inside the module declarations themselves, which means they can be moved to an archive or invoked exclusively via manual commands listed in the README. 【923cef†L1-L5】
- **Legacy invariance infrastructure.** The current training utilities still reach into `invariant_hedging.legacy` via `core.utils` and `modules.diagnostics`, so every consumer of modern APIs gets the entire legacy tree even though the majority of files (legacy objectives, report assets, diagnostics exports) are unused by the Hydra entrypoints. 【F:src/invariant_hedging/core/utils/__init__.py†L1-L15】【F:src/invariant_hedging/modules/diagnostics.py†L1-L19】
- **Real-anchor data loader cycle.** `tests/data/test_real_anchors.py` depends on `invariant_hedging.modules.data.real.loader`, but `core.utils` re-exports `legacy.utils.configs`, which imports the same loader, so the module fails to initialize. None of the runtime CLIs exercise this path, so the entire file is currently dead outside the failing test. 【F:src/invariant_hedging/legacy/utils/configs.py†L1-L74】【F:src/invariant_hedging/modules/data/real/loader.py†L1-L71】【F:tests/data/test_real_anchors.py†L1-L60】
- **Hydra quick-run stub.** `tools/quick_run.py` still contains the old `legacy.train.loop` CLI pasted below the current helper code, leaving two `if __name__ == "__main__"` blocks in the same file and a syntax error that prevents any of the `configs/examples/*` presets from running. 【F:tools/quick_run.py†L503-L550】
- **Diagnostics aggregation script.** `tools/scripts/compute_diagnostics.py` was truncated mid-function; the file starts with indented statements and never defines `_collect_diagnostics`, so it is unusable and currently only surfaces through lint errors. 【F:tools/scripts/compute_diagnostics.py†L1-L80】

### Unused or broken configs
- **Legacy algorithm/training/evaluation stacks.** The directories `configs/algorithm/`, `configs/training/`, `configs/evaluation/`, `configs/reproduce.yaml`, and `configs/logging/wandb.yaml` are only referenced by the deprecated quick-run/reproduce stack; no make target wires them into the current Hydra entrypoints. 【F:configs/reproduce.yaml†L1-L18】【F:configs/experiment_eval.yaml†L1-L18】
- **Example presets.** `configs/examples/hirm_minimal.yaml`, `configs/examples/real_anchors.yaml`, and `configs/examples/sim_crisis.yaml` point to the removed `hirm_head` objective and the broken quick-run driver, so they cannot be loaded. 【F:configs/examples/hirm_minimal.yaml†L1-L25】【F:tools/quick_run.py†L503-L550】
- **Real-anchor legacy data wiring.** `configs/data/real_anchor.yaml`, `configs/data/regimes.yaml`, and the `configs/run/plan_run*.yaml` files document an older release process but are not consumed anywhere in the modern Hydra workflows. 【F:configs/data/real_anchor.yaml†L1-L16】
- **Phase-2 and hirm_head objectives.** `configs/train/hirm_head.yaml`, `configs/train/phase2.yaml`, `configs/train/phase2_highlite.yaml`, and the matching `configs/model/hirm_head.yaml` survive only in documentation; `core.engine` now raises a warning when `objective == "hirm_head"`, so these files are dead weight. 【F:src/invariant_hedging/core/engine.py†L150-L178】
- **Config status table.** See [§1.4](#config-usage-and-status) for the exhaustive table covering all 87 YAML files.

### Potentially redundant tools/scripts
- **`tools/run_of_record.sh`.** The script hard-codes `train/<method>` and `eval/<window>` configs and mirrors runs into `reports/`; because `pip install -e .` currently fails, every invocation of `make paper` dies with `ModuleNotFoundError`, proving that the harness no longer functions on fresh clones. 【F:tools/run_of_record.sh†L1-L120】【914768†L1-L5】
- **`tools/scripts/make_scorecard.py`.** This script still references YAML-based checkpoints and legacy phase-2 table names, but nothing in the repository imports or executes it (the only matches are inside the file itself), so it is safe to archive. 【99e4f6†L1-L3】【F:tools/scripts/make_scorecard.py†L1-L40】
- **`tools/scripts/find_latest_checkpoint.py`.** This helper is the only script under `tools/scripts` still referenced (by `run_of_record.sh`). All other siblings (`diff_metrics.py`, `prepare_data.py`, `run_baseline.py`, etc.) are lint failures or prototypes with no call sites, so they should be pruned or moved into an archive directory after verification. 【F:tools/run_of_record.sh†L72-L140】【2439f3†L68-L97】

## 1.2 FUNCTION-LEVEL ANALYSIS

### Function-level dead code
The AST/usage scan surfaced 67 modules whose top-level functions or classes are never referenced anywhere else in the repo (including tests). Most live in legacy diagnostics/reporting or in visualization CLIs that only export plotting routines. These symbols should either move into an `archive/` namespace or be deleted to reduce maintenance burden.

### Per-module symbol listings
The table below lists every module that still contains zero-reference symbols and the specific functions/classes identified by the static analysis pass:

| Module | Zero-reference symbols |
| --- | --- |
| invariant_hedging.core.infra.io | write_sim_params_json (line 14), write_stress_summary_json (line 21) |
| invariant_hedging.evaluation.probes.spurious_vol | compute_msi_delta (line 59) |
| invariant_hedging.hirm.utils.determinism | resolve_seed (line 24) |
| invariant_hedging.legacy.diagnostics_v2.utils | safe_mean (line 76) |
| invariant_hedging.legacy.irm.head_grads | unfreeze_backbone (line 65) |
| invariant_hedging.legacy.models.heads | RiskHead (line 21) |
| invariant_hedging.legacy.objectives.entropic | entropic_risk_from_pnl (line 12) |
| invariant_hedging.legacy.objectives.invariance | collect_head_parameters (line 12) |
| invariant_hedging.legacy.utils.checkpoints | load_checkpoint (line 83) |
| invariant_hedging.legacy.utils.configs | environment_order (line 50) |
| invariant_hedging.legacy.utils.stats | bootstrap_mean_ci (line 43), qq_plot_data (line 33), turnover_ratio (line 29) |
| invariant_hedging.modules.data.real.anchors | episode_index_frame (line 148), validate_non_overlapping (line 91) |
| invariant_hedging.modules.data.real.loader | load_real_anchors (line 239) |
| invariant_hedging.modules.data.real_spy | load_real_dataset (line 62) |
| invariant_hedging.modules.markets.costs | apply_transaction_costs (line 20) |
| invariant_hedging.modules.markets.pricing | black_scholes_vega (line 91) |
| tests.data.test_real_anchors | test_anchor_boundaries_respected (line 95), test_episode_determinism (line 45), test_missing_options_graceful (line 108), test_no_overlap_across_splits (line 58), test_tagged_output_directories (line 79) |
| tests.diagnostics.test_export_schema | test_export_csv_schema_and_manifest (line 22) |
| tests.diagnostics.test_invariance | test_components_match_formulas (line 9), test_invariance_gap_computation (line 86), test_trimmed_mean_keeps_middle_mass (line 38), test_weighted_aggregation_matches_config (line 59) |
| tests.diagnostics.test_isi_deterministic | test_c1_global_stability_monotonic (line 14), test_c2_mechanistic_alignment_cases (line 26), test_c3_structural_stability_distance (line 36), test_isi_aggregation_matches_weights (line 54) |
| tests.diagnostics.test_metrics_monotonicity | test_ig_non_negative (line 25), test_turnover_non_negative_and_zero_when_constant (line 32), test_variance_increases_with_dispersion (line 16), test_worst_group_matches_max (line 9) |
| tests.diagnostics.test_robustness_efficiency | test_er_and_tr_formulas (line 42), test_vr_uses_mean_and_std (line 27), test_wg_matches_cvar_objective (line 13) |
| tests.envs.test_registry_real | test_registry_returns_expected_names (line 42), test_registry_unknown_name (line 51) |
| tests.envs.test_registry_synth | test_registry_contains_expected_regimes (line 25), test_tags_reflect_stress_flags (line 40) |
| tests.figures.test_make_all_figures | test_make_all_figures_end_to_end (line 29) |
| tests.figures.test_plot_functions | test_individual_plot_scripts (line 21) |
| tests.figures.test_table_readers | test_alignment_head_alias_columns (line 172), test_capital_efficiency_alias_columns (line 85), test_diagnostics_summary_alias_columns (line 131), test_read_alignment_head (line 154), test_read_capital_efficiency_frontier (line 66), test_read_diagnostics_summary_filters (line 104), test_read_invariance_diagnostics_success (line 23), test_read_invariance_missing_column (line 45) |
| tests.irm.test_head_only_integration | test_compute_env_head_grads_returns_flat_vectors (line 42), test_freeze_backbone_only_affects_phi (line 33), test_penalties_reflect_alignment_behaviour (line 55), test_training_step_combines_base_and_penalty (line 78) |
| tests.irm.test_penalties_unit | test_cosine_penalty_handles_zero_gradients (line 36), test_cosine_penalty_invariant_to_scaling (line 27), test_cosine_penalty_large_for_opposite_grads (line 13), test_cosine_penalty_matches_pairwise_average (line 19), test_cosine_penalty_zero_for_identical_grads (line 7), test_varnorm_handles_zero_gradients (line 67), test_varnorm_increases_with_gradient_noise (line 51), test_varnorm_invariant_to_scaling (line 58), test_varnorm_zero_for_identical_normalised_grads (line 44) |
| tests.report.test_aggregation_small | test_cross_seed_aggregation (line 28) |
| tests.report.test_ire3d_qc | test_ire3d_coordinates_and_assets (line 44) |
| tests.report.test_latex_tables | test_build_table_with_booktabs (line 10) |
| tests.report.test_plots_smoke | test_plotting_smoke (line 51) |
| tests.scripts.test_aggregate_diagnostics | test_aggregate_diagnostics_creates_tables (line 24) |
| tests.sim.test_calibration_moments | test_moments_calibration_ci_suite (line 132), test_moments_calibration_heavy_suite (line 137) |
| tests.sim.test_liquidity_stress | test_cost_monotonicity (line 10), test_costs_are_non_negative_and_units_correct (line 58), test_liquidity_applies_only_to_configured_regimes (line 21) |
| tests.sim.test_merton_overlay | test_disabled_overlay_returns_identical_path (line 39), test_left_tail_thickens_with_jumps (line 25), test_overlay_is_reproducible (line 16) |
| tests.sim.test_pricing_sanity | test_pricing_monotonicity_calm_to_crisis (line 71), test_sabr_crisis_widens_smile (line 112) |
| tests.test_checkpoint_manager | test_checkpoint_manager_prunes_missing_and_excess_checkpoints (line 24), test_checkpoint_manager_restores_existing_entries (line 14) |
| tests.test_costs | test_execution_cost_linear_quadratic (line 6) |
| tests.test_cvar_estimator | test_bootstrap_ci_has_width (line 18), test_cvar_matches_numpy_quantile (line 6), test_cvar_scales_down_with_losses (line 25) |
| tests.test_data_pipeline | test_feature_engineer_exposes_base_feature_names (line 15), test_unwrap_experiment_config_returns_train_block (line 8) |
| tests.test_determinism | test_smoke_metrics_repeatable (line 34) |
| tests.test_diagnostics | test_safe_eval_metric_average_is_float (line 14), test_safe_eval_metric_handles_missing_key (line 8) |
| tests.test_diagnostics_eval_only | test_gradients_ignore_diagnostics (line 73), test_hirm_loss_diagnostics_detached (line 81) |
| tests.test_erm_base_regression | test_erm_base_crisis_metrics_stable (line 53) |
| tests.test_eval_baselines | test_evaluate_baselines_produces_risk_metrics (line 34) |
| tests.test_feature_engineer_env | test_feature_engineer_fit_and_transform_matches_manual_scaling (line 34), test_feature_engineer_transform_requires_fit (line 49), test_feature_groups_respects_configuration_overrides (line 58), test_single_asset_env_simulation_tracks_basic_metrics (line 88) |
| tests.test_grad_align | test_env_variance_matches_tensor_var (line 36), test_normalized_head_grads_preserve_shape_and_scale (line 6), test_pairwise_cosine_alignment_monotonicity (line 22) |
| tests.test_gradient_penalty | test_hirm_penalty_alignment_and_variance (line 43), test_hirm_penalty_single_environment_degenerates (line 73), test_irm_penalty_matches_polynomial_reference (line 11), test_vrex_penalty_variance_gradient (line 29) |
| tests.test_hgca_penalty | test_cosine_alignment_penalty_is_positive_for_opposing_vectors (line 14), test_cosine_alignment_penalty_is_zero_for_aligned_vectors (line 8) |
| tests.test_hirm_head | test_alignment_logging_creates_csv (line 161), test_alignment_writer_appends (line 189), test_cosine_penalty_values (line 116), test_freeze_phi_blocks_gradients (line 68), test_lambda_zero_matches_mean_risk (line 103), test_normalization_invariance (line 140) |
| tests.test_ig | test_ig_empty_mapping_returns_nan (line 13), test_ig_ignores_empty_tensors (line 43), test_ig_mixed_scalars_and_tensors (line 29), test_ig_single_environment_returns_zero (line 20) |
| tests.test_isi | test_c1_extreme_dispersion_clamps_to_zero (line 26), test_c1_single_environment_is_perfect (line 18), test_c2_mechanistic_alignment_handles_opposites (line 37), test_c3_structural_stability_extreme_distance (line 45), test_isi_rejects_invalid_weight_lengths (line 67), test_isi_weighted_average_respects_bounds (line 57) |
| tests.test_paper_provenance | test_collect_provenance_handles_missing_run (line 40), test_collect_provenance_includes_expected_fields (line 25), test_write_provenance_roundtrip (line 46) |
| tests.test_penalties | test_irm_penalty_reduces_variance (line 9), test_vrex_penalty_reduces_variance (line 29) |
| tests.test_pricing_greeks | test_black_scholes_delta_matches_autograd (line 10), test_black_scholes_gamma_matches_autograd (line 22) |
| tests.test_real_windows | test_each_window_produces_expected_range (line 11) |
| tests.test_report_schema | test_invalid_payloads_raise (line 55), test_nested_schema_round_trip (line 24), test_sample_final_metrics_file_validates (line 15) |
| tests.test_reproduce | test_paper_provenance_placeholder_exists (line 13), test_run_of_record_writes_to_reports_tree (line 6) |
| tests.test_seed_reproducibility | test_final_metrics_match (line 37) |
| tests.test_seeding | test_seed_everything_deterministic (line 8) |
| tests.test_sim_regimes | test_feature_toggles_adjust_outputs (line 33), test_regime_schedule_is_deterministic (line 25) |
| tests.test_spurious_probe | test_spurious_probe_amplify_increases_msi_and_limits_hirm_delta (line 10) |
| tests.test_spy_loader | test_cli_smoke (line 112), test_load_cboe_series_accepts_cboe_headers (line 49), test_real_spy_module_train_val_test_splits (line 155), test_slice_inclusive_bounds (line 65), test_validation_overlap_detection (line 84) |
| tests.test_vr | test_vr_empty_mapping_returns_nan (line 13), test_vr_handles_identical_risks (line 43), test_vr_matches_population_variance (line 29), test_vr_single_environment_returns_zero (line 20) |
| tools.scripts.make_scorecard | _find_latest_checkpoint (line 573) |
## 1.3 IMPORT GRAPH & CYCLES
- **Scale of the graph.** The source tree exposes 137 Python modules under `src/invariant_hedging`. Running Tarjan’s SCC pass over the import adjacency list found *no* cycles, but 37 modules have zero inbound edges, meaning they are never imported by any other module in `src/`. These are primarily visualization CLIs (`plot_invariance_vs_ig`, `plot_regime_panels`, etc.), orphaned infrastructure (`core.infra.paths`, `core.metrics`), and legacy report utilities. 【891890†L1-L37】【F:src/invariant_hedging/core/infra/paths.py†L1-L34】
- **Cross-package leakage.** The `core` and `modules` packages both import from `invariant_hedging.legacy`, so even modern training runs pull the entire legacy tree into the runtime. For example, `core.utils` re-exports `legacy.utils.checkpoints/logging/seed/stats`, and `modules.diagnostics` re-uses the legacy ISI/IG computations, preventing us from isolating legacy code. 【F:src/invariant_hedging/core/utils/__init__.py†L1-L15】【F:src/invariant_hedging/modules/diagnostics.py†L1-L19】
- **Hydra entrypoints.** Only two modules declare `@hydra.main`: `core.engine.main` for training and `evaluation.evaluate_crisis.main` for diagnostics/evaluation. Both hard-code `config_path=str(get_repo_root()/"configs")`, so the entire configs tree is implicitly part of the import graph even though a large fraction of YAML files are unused. 【F:src/invariant_hedging/core/engine.py†L445-L471】【F:src/invariant_hedging/evaluation/evaluate_crisis.py†L680-L744】

## 1.4 CONFIG USAGE AND STATUS
The following table consolidates the status of every YAML file under `configs/`. “Core” configs are used by `tools/run_train.sh`, `tools/run_eval.sh`, `make paper`, or the pytest suite; “Auxiliary” configs drive smoke/sanity/reporting flows; “Broken” configs reference the deprecated quick-run/reproduce stack; and “Unused” configs are not referenced anywhere in the repository. Command usage is derived from the Makefile targets (`make paper`, `make report-paper`, `make real`, etc.) and the release harness. 【F:Makefile†L1-L115】【F:tools/run_of_record.sh†L1-L140】

| Config | Status |
| --- | --- |
| configs/algorithm/erm.yaml | Unused – part of the legacy configs/reproduce algorithm suite. |
| configs/algorithm/erm_reg.yaml | Unused – part of the legacy configs/reproduce algorithm suite. |
| configs/algorithm/groupdro.yaml | Unused – part of the legacy configs/reproduce algorithm suite. |
| configs/algorithm/irm.yaml | Unused – part of the legacy configs/reproduce algorithm suite. |
| configs/algorithm/vrex.yaml | Unused – part of the legacy configs/reproduce algorithm suite. |
| configs/base.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/costs/crisis.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/costs/high.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/costs/low.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/data/real_anchor.yaml | Unused – superseded by invariant_hedging.modules.data.real loader. |
| configs/data/real_spy.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/data/real_spy_paper.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/data/regimes.yaml | Unused – leftover dataset registry for the removed legacy runner. |
| configs/data/synthetic.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/diagnostics.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/diagnostics/default.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/envs/crisis.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/envs/daily.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/envs/high.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/envs/high_eval.yaml | Unused – no defaults or CLIs select this regime. |
| configs/envs/high_slim.yaml | Unused – only listed in unused phase2_highlite config. |
| configs/envs/jumps/merton_crisis.yaml | Unused – replaced by configs/sim/merton_crisis in tests. |
| configs/envs/liquidity/high_spread.yaml | Unused – never selected after search. |
| configs/envs/low.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/envs/med.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/eval/daily.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/eval/high.yaml | Unused – no references in the current CLI, tests, or make targets. |
| configs/eval/paper.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/eval/probes/spurious_vol.yaml | Unused – no references in the current CLI, tests, or make targets. |
| configs/eval/robustness.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/eval/smoke.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/evaluation/default.yaml | Unused – only referenced by configs/reproduce. |
| configs/examples/hirm_minimal.yaml | Broken – references the deprecated quick_run/reproduce stack. |
| configs/examples/real_anchors.yaml | Broken – references the deprecated quick_run/reproduce stack. |
| configs/examples/sim_crisis.yaml | Broken – references the deprecated quick_run/reproduce stack. |
| configs/experiment.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/experiment_eval.yaml | Broken – references the deprecated quick_run/reproduce stack. |
| configs/logging/default.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/logging/wandb.yaml | Unused – only referenced by configs/reproduce. |
| configs/model/delta.yaml | Unused – delta baseline instantiated programmatically during evaluation. |
| configs/model/delta_gamma.yaml | Unused – delta-gamma baseline instantiated programmatically during evaluation. |
| configs/model/erm.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/model/erm_reg.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/model/groupdro.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/model/hedger.yaml | Unused – only referenced by configs/reproduce. |
| configs/model/hirm_head.yaml | Unused – legacy hirm_head objective is deprecated in the engine. |
| configs/model/irm.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/model/vrex.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/paper/data.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/paper/eval.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/paper/methods.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/paper/train.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/report/default.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/report/paper.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/reproduce.yaml | Broken – references the deprecated quick_run/reproduce stack. |
| configs/run/plan_run1.yaml | Unused – documentation artifact for old paper runs. |
| configs/run/plan_run2.yaml | Unused – documentation artifact for old paper runs. |
| configs/sanity/erm_sanity.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sanity/hirm_sanity.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sim/heston_calm.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sim/heston_crisis.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sim/liquidity_calm.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sim/liquidity_crisis.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sim/merton_calm.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sim/merton_crisis.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sim/sabr_calm.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/sim/sabr_crisis.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/splits/spy_test_2008.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/splits/spy_test_2018.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/splits/spy_test_2020.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/splits/spy_test_2022.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/splits/spy_train.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/splits/spy_val.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/train/erm.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/train/erm_reg.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/train/examples/irm_cosine.yaml | Unused – prototype override never referenced. |
| configs/train/examples/irm_varnorm.yaml | Unused – prototype override never referenced. |
| configs/train/groupdro.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/train/hirm_head.yaml | Unused – deprecated objective; guarded against in core.engine. |
| configs/train/irm.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/train/paper.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/train/phase2.yaml | Unused – referenced only in legacy phase-2 notes. |
| configs/train/phase2_highlite.yaml | Unused – depends on hirm_head legacy objective. |
| configs/train/smoke.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/train/smoke_override.yaml | Auxiliary but used – required for smoke tests, sanity scripts, or paper packaging. |
| configs/train/vrex.yaml | Core production – consumed by the Hydra train/eval entrypoints. |
| configs/training/default.yaml | Unused – no references in the current CLI, tests, or make targets. |
## PHASE 2 PROPOSAL – CLEAN STRUCTURE FOR HIRM

### Target folder layout
```
src/invariant_hedging/
  cli/              # canonical Hydra entrypoints (train.py, eval.py, report.py)
  data/             # data modules (real loaders, feature engineering, env registries)
  training/         # optimizer/objective/engine implementations
  diagnostics/      # invariance/robustness diagnostics shared by train+eval
  evaluation/       # crisis evaluation harness + baseline adapters
  reporting/        # aggregation + plotting + latex/report generation
  visualization/    # thin wrappers that depend on reporting outputs
  runtime/          # shared utilities (device resolution, logging, checkpoints, seeding)
  legacy/           # archived Phase-1/Phase-2 experiments kept read-only
```
- Move everything under `src/invariant_hedging/core/` into `training/` (engine, optimizers, losses) and `runtime/` (device/logging). This removes the ambiguous “core” naming and makes the import graph explicit.
- Promote `src/invariant_hedging/modules/data_pipeline.py`, `modules/data/*`, and `modules/environment.py` into `data/` and `runtime/` so that data-prep utilities no longer sit alongside objectives/diagnostics.
- Collapse `evaluation/reporting` and `tools/report/*` into `reporting/` so that tables/figures/CLI live together. `tools/report/generate_report.py` becomes `src/invariant_hedging/reporting/cli.py`.
- Keep `visualization/` as thin wrappers around report artifacts; anything that only draws figures from `reports/paper` stays here, otherwise the logic belongs in `reporting/`.
- Freeze `legacy/` by moving it to `archive/legacy/` and making imports explicit (no more `core.utils` re-exports). Any remaining callers must consciously import from `archive.legacy`.

### Mapping from current structure
| Current path | Proposed destination | Action |
| --- | --- | --- |
| `src/invariant_hedging/core/engine.py` | `src/invariant_hedging/training/engine.py` | Move + rename module; keep `@hydra.main` in `cli/train.py`. |
| `src/invariant_hedging/core/utils/__init__.py` | `src/invariant_hedging/runtime/{device,logging,checkpoint}.py` | Split into focused utilities and remove legacy re-exports. |
| `src/invariant_hedging/modules/data_pipeline.py` | `src/invariant_hedging/data/pipeline.py` | Move and co-locate with loaders/env registries. |
| `src/invariant_hedging/modules/diagnostics.py` | `src/invariant_hedging/diagnostics/invariance.py` | Move and rewrite to avoid `legacy` dependencies. |
| `src/invariant_hedging/evaluation/evaluate_crisis.py` | `src/invariant_hedging/evaluation/cli.py` | Keep Hydra entrypoint but slim down once diagnostics move out. |
| `src/invariant_hedging/evaluation/reporting/*` + `tools/report/generate_report.py` | `src/invariant_hedging/reporting/*` | Merge packages and expose a single CLI for `make report`/`make report-paper`. |
| `src/invariant_hedging/visualization/*.py` | `src/invariant_hedging/visualization/*.py` (unchanged) | Re-export only the CLIs that rely on finalized report assets; archive the rest. |
| `src/invariant_hedging/legacy/*` | `archive/legacy/*` | Move wholesale and update imports to reference the archive explicitly. |
| `tools/run_of_record.sh`, `tools/quick_run.py`, unused `tools/scripts/*.py` | Remove or move into `archive/tools/` | Replace with a single Python CLI under `cli/` that mirrors the paper harness. |

### Config cleanup plan
- **Prune** the entire `configs/algorithm/`, `configs/training/`, `configs/evaluation/`, `configs/examples/`, `configs/run/`, `configs/reproduce.yaml`, and `configs/logging/wandb.yaml` trees—they only serve the broken quick-run stack.
- **Consolidate** smoke/paper configs by introducing `configs/train/smoke.yaml` overrides via Hydra defaults rather than bespoke files; house everything under `configs/train/` and `configs/eval/` with consistent naming.
- **Namespace** reporting configs under `configs/report/` (keep `default` + `paper`) and document the mapping to `reporting/cli.py`.
- **Document** the handful of auxiliary configs (sanity, sim, splits) that are still consumed by tests/make targets so contributors know they remain live.

### Dead code disposition
- Delete `tools/quick_run.py`, `tools/scripts/compute_diagnostics.py`, `tools/scripts/make_scorecard.py`, `tools/scripts/diff_metrics.py`, `tools/scripts/prepare_data.py`, and `tools/scripts/run_baseline.py` after archiving any documentation—they currently fail lint and have no call sites.
- Remove the unused configs called out in §1.4 once the new folder layout lands.
- Archive the visualization CLIs that never run in CI (those listed in §1.2) under `archive/visualization/` or drop them entirely after confirming the README no longer references them.
- Keep the real-anchor tests disabled until the circular import is resolved, then reinstate them as part of the data regression suite.

This restructuring keeps the public API surface (`cli/train.py`, `cli/eval.py`, `cli/report.py`) minimal, isolates `legacy` dependencies, and gives each subsystem (data, training, evaluation, reporting) an obvious home for future refactors.
