# Training Objectives

## HIRM Head Alignment

The `hirm_head` objective constrains the decision head (ψ) by aligning its
per-environment risk gradients. Only gradients with respect to ψ contribute to
the alignment penalty, while the backbone (φ) can be optionally frozen via the
`model.freeze_phi` flag. Configuration lives in `config/hirm.yaml` and exposes:

- `hirm.lambda_align`: penalty strength (`0.0` disables alignment).
- `hirm.normalize_grad`: L2-normalise per-environment ψ gradients before
  computing pairwise similarities.
- `hirm.pairwise_mode`: choose between cosine misalignment (`cosine`) or
  variance of normalised gradients (`var`).
- `hirm.detach_features`: detach representation features before computing ψ
  gradients, preventing penalty gradients from flowing into φ.

See `docs/devnotes/hirm_head.md` for implementation details and gradient flow.
