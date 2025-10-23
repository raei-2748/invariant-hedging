"""Command-line entrypoint for the reporting aggregation pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.evaluation.reporting.aggregate import (
    AggregateResult,
    aggregate_runs,
    build_table_dataframe,
    load_report_config,
)
from src.evaluation.reporting.ire3d import build_ire_coordinates, write_ire_assets
from src.evaluation.reporting.latex import build_table, save_latex_table, write_table_csv
from src.evaluation.reporting.plots import (
    plot_efficiency_frontier,
    plot_heatmaps,
    plot_qq,
    plot_scorecard,
    plot_seed_distributions,
)
from src.evaluation.reporting.provenance import build_manifest, write_manifest


def _ensure_directories(base: Path) -> Dict[str, Path]:
    paths = {
        "tables": base / "tables",
        "figures": base / "figures",
        "interactive": base / "interactive",
        "manifests": base / "manifests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_tables(result: AggregateResult, output_dirs: Dict[str, Path]) -> None:
    report_cfg = result.config["report"]
    regimes = report_cfg.get("regimes_order", result.regimes)
    metrics_cfg = report_cfg.get("metrics", {})

    blocks = [
        ("scorecard", [metric for metrics in metrics_cfg.values() for metric in metrics], "Main scorecard"),
        ("invariance", metrics_cfg.get("invariance", []), "Invariance metrics"),
        ("robustness", metrics_cfg.get("robustness", []), "Robustness metrics"),
        ("efficiency", metrics_cfg.get("efficiency", []), "Efficiency metrics"),
    ]
    for table_name, metrics, caption in blocks:
        if not metrics:
            continue
        table_text = build_table(
            result.summary,
            metrics=metrics,
            regimes=regimes,
            config=report_cfg,
            caption=caption,
            label=f"tab:{table_name}",
        )
        save_latex_table(table_text, output_dirs["tables"] / f"{table_name}.tex")
        pivot = build_table_dataframe(result.summary, metrics, regimes)
        write_table_csv(pivot, output_dirs["tables"] / f"{table_name}.csv")

    # Optional per-seed export for appendix
    raw_path = output_dirs["tables"] / "seed_values.csv"
    result.raw.to_csv(raw_path, index=False)


def _write_plots(result: AggregateResult, output_dirs: Dict[str, Path], lite: bool) -> None:
    report_cfg = result.config["report"]
    figures_dir = output_dirs["figures"]
    plot_scorecard(result.summary, report_cfg, figures_dir)
    plot_heatmaps(result.summary, report_cfg, figures_dir)

    robustness_metrics = report_cfg.get("metrics", {}).get("robustness", [])
    invariance_metrics = report_cfg.get("metrics", {}).get("invariance", [])
    efficiency_metrics = report_cfg.get("metrics", {}).get("efficiency", [])

    target_metric = None
    if efficiency_metrics:
        target_metric = efficiency_metrics[0]
    elif invariance_metrics:
        target_metric = invariance_metrics[0]
    if target_metric and not lite:
        plot_qq(result.raw, report_cfg, figures_dir, metric=target_metric)
    seed_metrics: list[str] = []
    if robustness_metrics:
        seed_metrics.append(robustness_metrics[0])
    if invariance_metrics:
        seed_metrics.append(invariance_metrics[0])
    if seed_metrics:
        plot_seed_distributions(result.raw, seed_metrics, report_cfg, figures_dir)
    if len(efficiency_metrics) >= 2:
        plot_efficiency_frontier(
            result.raw,
            efficiency_metric=efficiency_metrics[0],
            turnover_metric=efficiency_metrics[1],
            config=report_cfg,
            output_dir=figures_dir,
        )


def run_pipeline(args: argparse.Namespace) -> None:
    config = load_report_config(Path(args.config))
    report_cfg = config.get("report", {})
    if args.out:
        report_cfg["outputs_dir"] = args.out
        config["report"] = report_cfg
    if args.lite:
        report_cfg["seeds"] = min(int(report_cfg.get("seeds", 5)), 5)
    config["report"] = report_cfg
    outputs_dir = Path(report_cfg.get("outputs_dir", "outputs/report_assets"))
    dirs = _ensure_directories(outputs_dir)

    result = aggregate_runs(config, lite=args.lite)

    _write_tables(result, dirs)
    _write_plots(result, dirs, lite=args.lite)

    generate_3d = report_cfg.get("generate_3d", False) and not args.skip_3d
    extra_manifest: Dict[str, List[str] | Dict[str, List[float]] | float] = {}
    if generate_3d:
        ire_result = build_ire_coordinates(result.raw, config)
        points_path = dirs["tables"] / "ire_points.csv"
        ire_result.points.to_csv(points_path, index=False)
        write_ire_assets(ire_result.points, config, outputs_dir)
        extra_manifest.update(
            {
                "ire3d": {
                    "winsor_bounds": {k: list(v) for k, v in ire_result.winsor_bounds.items()},
                    "minmax_bounds": {k: list(v) for k, v in ire_result.minmax_bounds.items()},
                    "alpha": ire_result.alpha,
                }
            }
        )

    manifest = build_manifest(result, report_cfg, extra_manifest)
    write_manifest(manifest, dirs["manifests"] / "aggregate_manifest.json")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate 30-seed diagnostics into report assets")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    parser.add_argument("--lite", action="store_true", help="Run in lite mode (≤5 seeds, lighter plots)")
    parser.add_argument("--out", help="Override outputs directory")
    parser.add_argument("--skip-3d", action="store_true", help="Skip IRE 3D rendering")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    run_pipeline(parse_args())
