import math

import numpy as np
import pandas as pd

from invariant_hedging.diagnostics.robustness import compute_IG, compute_invariance_spectrum


def test_components_match_formulas():
    risk_df = pd.DataFrame(
        [
            {"probe_id": "p0", "env": "A", "risk": -0.5},
            {"probe_id": "p0", "env": "B", "risk": 0.5},
        ]
    )
    grad_df = pd.DataFrame(
        [
            {"probe_id": "p0", "env_i": "A", "env_j": "B", "cosine": 0.2},
            {"probe_id": "p0", "env_i": "A", "env_j": "B", "cosine": 0.4},
        ]
    )
    rep_df = pd.DataFrame(
        [
            {"probe_id": "p0", "dispersion": 0.3},
        ]
    )

    cfg = {"trim_ratio": 0.0, "tau_risk": 1.0, "tau_cov": 1.0}
    result = compute_invariance_spectrum(rep_df, grad_df, risk_df, cfg)
    assert math.isclose(result["C1"], 0.75, rel_tol=1e-6)
    assert math.isclose(result["C2"], 0.65, rel_tol=1e-6)
    assert math.isclose(result["C3"], 0.7, rel_tol=1e-6)
    expected_isi = (0.75 + 0.65 + 0.7) / 3.0
    assert math.isclose(result["ISI"], expected_isi, rel_tol=1e-6)
    assert result["per_probe"]["p0"]["C1"] == result["C1"]


def test_trimmed_mean_keeps_middle_mass():
    probes = []
    for idx, variance in enumerate(np.linspace(0.0, 0.9, 10)):
        delta = math.sqrt(variance)
        probes.append({"probe_id": f"p{idx}", "env": "A", "risk": -delta})
        probes.append({"probe_id": f"p{idx}", "env": "B", "risk": delta})
    risk_df = pd.DataFrame(probes)
    grad_df = pd.DataFrame(columns=["probe_id", "env_i", "env_j", "cosine"])
    rep_df = pd.DataFrame(columns=["probe_id", "dispersion"])

    cfg = {"trim_ratio": 0.1, "tau_risk": 1.0, "weights": {"C1": 1.0}}
    result = compute_invariance_spectrum(rep_df, grad_df, risk_df, cfg)

    c1_values = [1.0 - variance for variance in np.linspace(0.0, 0.9, 10)]
    sorted_vals = sorted(c1_values)
    trimmed_vals = sorted_vals[1:-1]
    expected = sum(trimmed_vals) / len(trimmed_vals)
    assert math.isclose(result["C1"], expected, rel_tol=1e-6)
    assert math.isclose(result["ISI"], expected, rel_tol=1e-6)


def test_weighted_aggregation_matches_config():
    variance = 0.2
    delta = math.sqrt(variance)
    risk_df = pd.DataFrame(
        [
            {"probe_id": "p0", "env": "A", "risk": -delta},
            {"probe_id": "p0", "env": "B", "risk": delta},
        ]
    )
    grad_df = pd.DataFrame(
        [
            {"probe_id": "p0", "env_i": "A", "env_j": "B", "cosine": 0.2},
        ]
    )
    rep_df = pd.DataFrame(
        [
            {"probe_id": "p0", "dispersion": 0.6},
        ]
    )

    cfg = {"trim_ratio": 0.0, "tau_risk": 1.0, "tau_cov": 1.0, "weights": {"C1": 0.2, "C2": 0.3, "C3": 0.5}}
    result = compute_invariance_spectrum(rep_df, grad_df, risk_df, cfg)

    expected = 0.2 * 0.8 + 0.3 * 0.6 + 0.5 * 0.4
    assert math.isclose(result["ISI"], expected, rel_tol=1e-6)


def test_invariance_gap_computation():
    env_df = pd.DataFrame(
        [
            {"env": "A", "value": 0.2},
            {"env": "B", "value": 0.5},
            {"env": "C", "value": 0.1},
        ]
    )
    result = compute_IG(env_df, {"tau_norm": 2.0})
    assert math.isclose(result["IG"], 0.4, rel_tol=1e-6)
    assert math.isclose(result["IG_norm"], 0.2, rel_tol=1e-6)
