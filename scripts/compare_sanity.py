#!/usr/bin/env python3
"""Compare ERM vs. HIRM sanity metrics and emit a PASS/FAIL verdict."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

METHODS = ("erm", "hirm")
CRISIS_ENV = ("crisis", "test")
PASS_COLOR = "\033[32m"
FAIL_COLOR = "\033[31m"
RESET_COLOR = "\033[0m"


def _color(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{RESET_COLOR}"
    return text


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected metrics file: {path}")
    return json.loads(path.read_text())


def _load_crisis_row(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing smoke results CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("env"), row.get("split")) == CRISIS_ENV:
                return {k: float(v) for k, v in row.items() if v not in {None, ""}}
    raise RuntimeError(f"Unable to locate crisis row in {csv_path}")


def _extract_metrics(root: Path, method: str) -> Dict[str, float]:
    crisis_eval = root / method / "eval" / "crisis"
    final_metrics = _load_json(crisis_eval / "final_metrics.json")
    diag_summary = _load_json(crisis_eval / "artifacts" / "diagnostics_summary.json")
    crisis_row = _load_crisis_row(crisis_eval / "artifacts" / "smoke_results.csv")

    def require(key: str) -> float:
        if key not in final_metrics:
            raise KeyError(f"Metric '{key}' missing from {crisis_eval / 'final_metrics.json'}")
        return float(final_metrics[key])

    msi_block = diag_summary.get("MSI", {})
    isi_value = msi_block.get("value", {}).get("mean") if msi_block else None
    if isi_value is None:
        raise RuntimeError(
            "MSI/ISI diagnostics missing; ensure eval.compute_msi=true was set during evaluations."
        )

    metrics = {
        "method": method.upper(),
        "cvar95": require("test/crisis/ES95"),
        "ig": require("diagnostics/IG/ES95"),
        "isi": float(isi_value),
        "maxdd": float(crisis_row.get("max_drawdown")),
        "turnover": require("test/crisis/Turnover"),
    }
    return metrics


def _write_summary(root: Path, rows: List[Dict[str, float]], status: str, reasons: Dict[str, bool]) -> None:
    summary_dir = root
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "sanity_summary.json"
    csv_path = summary_dir / "sanity_summary.csv"

    payload = {"status": status, "comparisons": rows, "assertions": reasons}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_table(rows: List[Dict[str, float]]) -> str:
    headers = ["Method", "CVaR-95", "IG", "ISI", "MaxDD", "Turnover"]
    widths = {h: len(h) for h in headers}
    processed = []
    for row in rows:
        formatted = [
            row["method"],
            f"{row['cvar95']:.3f}",
            f"{row['ig']:.3f}",
            f"{row['isi']:.3f}",
            f"{row['maxdd']:.3f}",
            f"{row['turnover']:.3f}",
        ]
        for header, cell in zip(headers, formatted):
            widths[header] = max(widths[header], len(cell))
        processed.append(formatted)

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[h]) for cell, h in zip(cells, headers))

    lines = [fmt_row(headers), "-+-".join("-" * widths[h] for h in headers)]
    lines.extend(fmt_row(row) for row in processed)
    return "\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments/sanity", help="Sanity experiment root (default: %(default)s)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows: List[Dict[str, float]] = []
    metrics: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        metrics[method] = _extract_metrics(root, method)
        rows.append(metrics[method])

    cvar_rule = metrics["hirm"]["cvar95"] < metrics["erm"]["cvar95"]
    isi_rule = metrics["hirm"]["isi"] > metrics["erm"]["isi"]
    ig_rule = metrics["hirm"]["ig"] < metrics["erm"]["ig"]
    assertions = {
        "crisis_cvar_improves": cvar_rule,
        "isi_increases": isi_rule,
        "ig_decreases": ig_rule,
    }

    status = "PASS" if all(assertions.values()) else "FAIL"
    _write_summary(root, rows, status, assertions)

    print(_format_table(rows))
    if status == "PASS":
        print(
            _color(
                "\n✅ Sanity test passed — HIRM improves crisis robustness.\nSafe to run full sweeps.",
                PASS_COLOR,
            )
        )
        return
    print(
        _color(
            "\n❌ Sanity test failed — unexpected robustness behavior.\n"
            "Please debug the invariance pipeline before running full sweeps.",
            FAIL_COLOR,
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
