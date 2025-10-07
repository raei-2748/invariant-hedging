import argparse
import csv
import json
import pathlib
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from .metrics import invariance_gap, mechanism_sensitivity_index, worst_group_gap


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _choose_candidate(candidates: List[Tuple[str, Any]], preferences: Iterable[str]) -> Any:
    if not candidates:
        return ""
    lower_pref = [pref.lower() for pref in preferences]
    for pref in lower_pref:
        for key, value in candidates:
            if pref in key.lower():
                return value
    return candidates[0][1]


def _extract_metric(metrics: Dict[str, Any], keywords: Iterable[str]) -> Any:
    for key in keywords:
        if key in metrics:
            return metrics[key]
    candidates = []
    lowered_keywords = [kw.lower() for kw in keywords]
    for key, value in metrics.items():
        norm_key = key.replace("/", "_").lower()
        if any(kw in norm_key for kw in lowered_keywords):
            candidates.append((key, value))
    if not candidates:
        return ""
    return candidates[0][1]


def _derive_diagnostics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    ig_candidates = [
        (key, value)
        for key, value in metrics.items()
        if key.lower().startswith("diagnostics/ig") and value is not None
    ]
    wg_candidates = [
        (key, value)
        for key, value in metrics.items()
        if key.lower().startswith("diagnostics/wg") and value is not None
    ]
    msi_candidates = [
        (key, value)
        for key, value in metrics.items()
        if key.lower().startswith("diagnostics/msi") and value is not None
    ]

    ig = _choose_candidate(ig_candidates, ["test", "val", "train"])
    wg = _choose_candidate(wg_candidates, ["test", "val", "train"])
    msi = _choose_candidate(msi_candidates, ["test", "val", "train"])

    if ig == "":
        train_risks = metrics.get("train/env_risk", {})
        if isinstance(train_risks, list):
            train_risks = {str(idx): value for idx, value in enumerate(train_risks)}
        if train_risks:
            ig = invariance_gap(train_risks)
    if wg == "":
        train_risks = metrics.get("train/env_risk", {})
        test_risks = metrics.get("test/env_risk", {})
        if isinstance(train_risks, list):
            train_risks = {str(idx): value for idx, value in enumerate(train_risks)}
        if isinstance(test_risks, list):
            test_risks = {str(idx): value for idx, value in enumerate(test_risks)}
        if train_risks or test_risks:
            wg = worst_group_gap(train_risks, test_risks)
    if msi == "":
        s_phi = metrics.get("sensitivities/s_phi")
        s_r = metrics.get("sensitivities/s_r")
        if s_phi is not None and s_r is not None:
            msi = mechanism_sensitivity_index(float(s_phi), float(s_r))
    return {"IG": ig, "WG": wg, "MSI": msi}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect diagnostics from run directories.")
    parser.add_argument("--runs", required=True, help="Directory containing run outputs")
    parser.add_argument("--out", required=True, help="Path to write diagnostics CSV")
    args = parser.parse_args()

    rows = []
    for path in pathlib.Path(args.runs).glob("**/final_metrics.json"):
        metrics = _load_json(path)
        config = _load_yaml(path.parent / "config.yaml")
        metadata = _load_json(path.parent / "metadata.json")

        diagnostics = _derive_diagnostics(metrics)
        method = metrics.get("method") or config.get("model", {}).get("name", "")
        seed = metrics.get("seed") or config.get("train", {}).get("seed", "")
        commit = (
            metrics.get("commit")
            or metadata.get("git_commit")
            or metadata.get("commit")
            or ""
        )

        rows.append(
            {
                "method": method,
                "seed": seed,
                "cvar95": _extract_metric(metrics, ["cvar95", "crisis_cvar", "cvar", "es95"]),
                "mean": _extract_metric(metrics, ["mean", "mean_pnl", "pnl_mean"]),
                "sortino": _extract_metric(metrics, ["sortino"]),
                "turnover": _extract_metric(metrics, ["turnover"]),
                "IG": diagnostics["IG"],
                "WG": diagnostics["WG"],
                "MSI": diagnostics["MSI"],
                "commit": commit,
            }
        )

    output_path = pathlib.Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        if rows:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            csv_file.write("")


if __name__ == "__main__":
    main()
