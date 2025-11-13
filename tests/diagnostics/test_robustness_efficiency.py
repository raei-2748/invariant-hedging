import math

import pandas as pd

from invariant_hedging.diagnostics.robustness import (
    compute_ER,
    compute_TR,
    compute_VR,
    compute_WG,
)


def test_wg_matches_cvar_objective():
    risk_df = pd.DataFrame(
        [
            {"env": "A", "risk": 0.1},
            {"env": "B", "risk": 0.2},
            {"env": "C", "risk": 0.3},
            {"env": "D", "risk": 0.4},
        ]
    )
    result = compute_WG(risk_df, {"alpha": 0.25})
    assert math.isclose(result["WG"], 0.4, rel_tol=1e-6)
    assert math.isclose(result["tau"], 0.3, rel_tol=1e-6)


def test_vr_uses_mean_and_std():
    series_df = pd.DataFrame(
        [
            {"step": 0, "risk": 0.2},
            {"step": 1, "risk": 0.3},
            {"step": 2, "risk": 0.4},
            {"step": 3, "risk": 0.5},
        ]
    )
    result = compute_VR(series_df, {"epsilon": 0.0})
    assert math.isclose(result["mean"], 0.35, rel_tol=1e-6)
    assert math.isclose(result["std"], 0.11180339887498948, rel_tol=1e-9)
    assert math.isclose(result["VR"], 0.31943828249996997, rel_tol=1e-9)


def test_er_and_tr_formulas():
    pnl_df = pd.DataFrame(
        [
            {"step": 0, "pnl": 0.1, "action_x": 0.0, "action_y": 0.0},
            {"step": 1, "pnl": 0.2, "action_x": 0.5, "action_y": 0.0},
            {"step": 2, "pnl": -0.2, "action_x": 0.0, "action_y": 0.5},
        ]
    )
    er = compute_ER(pnl_df[["step", "pnl"]], {"cvar_alpha": 0.05, "epsilon": 0.0})
    assert math.isclose(er["mean_pnl"], 1.0 / 30.0, rel_tol=1e-9)
    assert math.isclose(er["cvar"], -0.2, rel_tol=1e-9)
    assert math.isclose(er["ER"], 1.0 / 6.0, rel_tol=1e-9)

    tr = compute_TR(pnl_df[["step", "action_x", "action_y"]], {"epsilon": 0.0})
    assert math.isclose(tr["mean_position"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(tr["mean_turnover"], 0.6035533905932737, rel_tol=1e-9)
    assert math.isclose(tr["TR"], 1.8106601717798212, rel_tol=1e-9)
