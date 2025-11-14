import torch

from invariant_hedging.evaluation.probes.spurious_vol import (
    SpuriousVolConfig,
    apply_spurious_vol_probe,
)
from invariant_hedging.diagnostics.invariance import mechanistic_sensitivity


def test_spurious_probe_amplify_increases_msi_and_limits_hirm_delta():
    feature_names = ["delta", "gamma", "time_to_maturity", "realized_vol", "inventory"]
    features_erm = torch.ones(10, 5)
    features_hirm = torch.full((10, 5), 0.5)

    base_config = SpuriousVolConfig(enabled=False)
    amplify_config = SpuriousVolConfig(enabled=True, mode="amplify", k=3.0)

    base_erm = apply_spurious_vol_probe(features_erm, feature_names, base_config)
    base_hirm = apply_spurious_vol_probe(features_hirm, feature_names, base_config)
    probed_erm = apply_spurious_vol_probe(features_erm, feature_names, amplify_config)
    probed_hirm = apply_spurious_vol_probe(features_hirm, feature_names, amplify_config)

    idx = feature_names.index("realized_vol")
    base_msi = mechanistic_sensitivity([base_erm[:, idx].abs().mean().item()])
    probed_msi = mechanistic_sensitivity([probed_erm[:, idx].abs().mean().item()])
    assert probed_msi > base_msi, "Amplifying realized vol should increase MSI"

    base_cvar_erm = base_erm[:, idx].mean().item()
    base_cvar_hirm = base_hirm[:, idx].mean().item()
    probed_cvar_erm = probed_erm[:, idx].mean().item()
    probed_cvar_hirm = probed_hirm[:, idx].mean().item()

    delta_erm = abs(probed_cvar_erm - base_cvar_erm)
    delta_hirm = abs(probed_cvar_hirm - base_cvar_hirm)
    assert delta_erm >= delta_hirm, "HIRM should degrade less than ERM under spurious perturbations"
