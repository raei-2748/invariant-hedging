#!/usr/bin/env python3
"""Generate paper-aligned report assets from aggregated diagnostics."""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (SRC_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from src.legacy.report_assets import attach_deltas, load_report_inputs
from src.legacy.report_assets.figures import (
    plot_capital_frontier,
    plot_cross_regime_heatmap,
    plot_cvar_violin,
    plot_head_vs_feature_ablation,
    plot_ig_vs_cvar,
    plot_penalty_sweep,
    plot_isi_decomposition,
)
from src.evaluation.reporting.aggregate import AggregateResult, aggregate_runs, build_table_dataframe, load_report_config
from src.evaluation.reporting.latex import build_table, save_latex_table, write_table_csv
from src.evaluation.reporting.provenance import build_manifest, write_manifest
from src.evaluation.reporting.schema import FinalMetricsValidationError, load_final_metrics

LOGGER = logging.getLogger("report.generate")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _ensure_directories(root: Path) -> Mapping[str, Path]:
    paths: MutableMapping[str, Path] = {
        "tables": root / "tables",
        "figures": root / "figures",
        "manifests": root / "manifests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_placeholder_latex(path: Path, caption: str) -> None:
    text = (
        "% Placeholder table generated in smoke mode\n"
        "\\begin{tabular}{l}\n"
        "\\toprule\\\\\n"
        f"{caption}\\\\\n"
        "\\midrule\\\\\n"
        "No data available\\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_placeholder_csv(path: Path, headers: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        handle.write(",".join(["placeholder" for _ in headers]) + "\n")


def _write_placeholder_figure(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, f"No data available\n({title})", ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _placeholder_result(config: Mapping[str, object]) -> AggregateResult:
    empty_raw = pd.DataFrame(columns=["regime", "metric", "value", "seed", "run_path"])
    empty_summary = pd.DataFrame(
        columns=[
            "metric",
            "regime",
            "block",
            "mean",
            "std",
            "ci_half_width",
            "ci_low",
            "ci_high",
            "min",
            "max",
            "n",
        ]
    )
    return AggregateResult(
        raw=empty_raw,
        summary=empty_summary,
        regimes=[],
        selected_seeds=[],
        config=dict(config),
    )


def _generate_tables(result: AggregateResult, tables_dir: Path, smoke: bool) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_cfg = result.config.get("report", {}) if isinstance(result.config, Mapping) else {}
    regimes = report_cfg.get("regimes_order", result.regimes)
    metrics_cfg = report_cfg.get("metrics", {}) if isinstance(report_cfg, Mapping) else {}
    blocks = [
        ("scorecard", [metric for metrics in metrics_cfg.values() for metric in metrics], "Main scorecard"),
        ("invariance", metrics_cfg.get("invariance", []), "Invariance metrics"),
        ("robustness", metrics_cfg.get("robustness", []), "Robustness metrics"),
        ("efficiency", metrics_cfg.get("efficiency", []), "Efficiency metrics"),
    ]

    generated: list[Path] = []
    for table_name, metrics, caption in blocks:
        tex_path = tables_dir / f"{table_name}.tex"
        csv_path = tables_dir / f"{table_name}.csv"
        if metrics and not result.summary.empty:
            table_text = build_table(
                result.summary,
                metrics=metrics,
                regimes=regimes if regimes else result.regimes,
                config=report_cfg,
                caption=caption,
                label=f"tab:{table_name}",
            )
            save_latex_table(table_text, tex_path)
            pivot = build_table_dataframe(result.summary, metrics, regimes if regimes else result.regimes)
            write_table_csv(pivot, csv_path)
            generated.extend([tex_path, csv_path])
        elif smoke:
            LOGGER.debug("Writing placeholder table for %s", table_name)
            _write_placeholder_latex(tex_path, caption)
            headers = ["metric", "note"]
            _write_placeholder_csv(csv_path, headers)
            generated.extend([tex_path, csv_path])
    raw_path = tables_dir / "seed_values.csv"
    if not result.raw.empty:
        result.raw.to_csv(raw_path, index=False)
        generated.append(raw_path)
    elif smoke:
        _write_placeholder_csv(raw_path, ["regime", "metric", "value"])
        generated.append(raw_path)
    return generated


def _generate_figures(
    per_seed_paths: Iterable[Path] | None,
    figures_dir: Path,
    smoke: bool,
    dpi: int,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    per_seed_paths = tuple(Path(p) for p in per_seed_paths) if per_seed_paths else None
    inputs = load_report_inputs(per_seed_paths=per_seed_paths)
    scorecard = attach_deltas(inputs.scorecard) if inputs.scorecard is not None else None
    generators = [
        ("fig_penalty_sweep.png", lambda path: plot_penalty_sweep(inputs, path, dpi)),
        ("fig_head_vs_feature.png", lambda path: plot_head_vs_feature_ablation(inputs, path, dpi)),
        ("fig_isi_decomposition.png", lambda path: plot_isi_decomposition(inputs, path, dpi)),
        ("fig_cross_regime_heatmap.png", lambda path: plot_cross_regime_heatmap(inputs, path, dpi)),
        ("fig_capital_frontier.png", lambda path: plot_capital_frontier(scorecard, path, dpi)),
        ("fig_cvar_violin.png", lambda path: plot_cvar_violin(inputs.diagnostics, scorecard, path, dpi)),
        ("fig_ig_vs_cvar.png", lambda path: plot_ig_vs_cvar(inputs.diagnostics, path, dpi)),
    ]

    produced: list[Path] = []
    for filename, fn in generators:
        out_path = figures_dir / filename
        try:
            success = fn(out_path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to generate %s: %s", filename, exc)
            success = False
        if success:
            produced.append(out_path)
        elif smoke:
            _write_placeholder_figure(out_path, filename)
            produced.append(out_path)
    return produced


def _discover_final_metrics(seed_dirs: Iterable[str]) -> list[Path]:
    candidates: list[Path] = []
    for pattern in seed_dirs:
        for raw in glob.glob(pattern):
            run_dir = Path(raw)
            metrics_path = run_dir / "final_metrics.json"
            if metrics_path.exists():
                candidates.append(metrics_path)
    unique = sorted({path.resolve() for path in candidates})
    return [Path(path) for path in unique]


def _validate_final_metrics(paths: Iterable[Path], smoke: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path in paths:
        entry: dict[str, object] = {"path": str(path)}
        if not path.exists():
            entry["status"] = "missing"
            results.append(entry)
            LOGGER.warning("Missing final_metrics.json at %s", path)
            if not smoke:
                raise FileNotFoundError(path)
            continue
        try:
            payload = load_final_metrics(path)
        except FinalMetricsValidationError as exc:
            entry["status"] = "invalid"
            entry["error"] = str(exc)
            results.append(entry)
            LOGGER.error("Invalid final_metrics payload at %s: %s", path, exc)
            if not smoke:
                raise
            continue
        entry.update(
            {
                "status": "ok",
                "schema_version": payload.schema_version,
                "metric_count": len(payload.metrics),
            }
        )
        results.append(entry)
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready report assets")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/report/default.yaml"),
        help="Report aggregation configuration",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for paper assets",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode (lenient validation, placeholder artefacts)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    config = load_report_config(Path(args.config))
    report_cfg = dict(config.get("report", {}))
    outputs_root = Path(args.out) if args.out else Path(report_cfg.get("paper_outputs_dir", "outputs/report_paper"))
    outputs_root.mkdir(parents=True, exist_ok=True)
    directories = _ensure_directories(outputs_root)

    if args.smoke:
        seeds = int(report_cfg.get("seeds", 5) or 5)
        report_cfg["seeds"] = min(seeds, 5)
    config["report"] = report_cfg

    try:
        result = aggregate_runs(config, lite=args.smoke)
    except Exception as exc:
        if not args.smoke:
            raise
        LOGGER.warning("Aggregation failed in smoke mode: %s", exc)
        result = _placeholder_result(config)

    tables = _generate_tables(result, directories["tables"], smoke=args.smoke)
    per_seed_paths = [sel.diagnostics_path for sel in getattr(result, "selected_seeds", []) if Path(sel.diagnostics_path).exists()]
    figures = _generate_figures(per_seed_paths or None, directories["figures"], smoke=args.smoke, dpi=args.dpi)

    if result.selected_seeds:
        final_metrics_paths = [sel.run_dir / "final_metrics.json" for sel in result.selected_seeds]
    else:
        seed_dirs_raw = report_cfg.get("seed_dirs", ["reports/artifacts/*"])
        if isinstance(seed_dirs_raw, str):
            seed_patterns = [seed_dirs_raw]
        elif isinstance(seed_dirs_raw, Iterable):
            seed_patterns = list(seed_dirs_raw)
        else:
            seed_patterns = []
        final_metrics_paths = _discover_final_metrics(seed_patterns)
    validations = _validate_final_metrics(final_metrics_paths, smoke=args.smoke)

    manifest_extra: dict[str, object] = {
        "smoke_mode": bool(args.smoke),
        "tables": [str(path.relative_to(outputs_root)) for path in tables],
        "figures": [str(path.relative_to(outputs_root)) for path in figures],
        "final_metrics": validations,
    }
    manifest = build_manifest(result, report_cfg, manifest_extra)
    manifest_path = directories["manifests"] / "paper_manifest.json"
    write_manifest(manifest, manifest_path)
    LOGGER.info("Provenance manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
