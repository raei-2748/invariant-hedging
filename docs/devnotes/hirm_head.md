# DevNote — HIRM Head Alignment

This objective aligns per-environment risk gradients with respect to the
decision head (ψ) while keeping the representation (φ) free to adapt or frozen
depending on the experiment.

```
features ──► φ (backbone) ──► representation ──┐
                                             │
                                             ▼
                                   ψ (decision head)
                                             │
                                             ▼
                                      risk / loss
```

1. Compute per-environment risk losses `R_e`.
2. Re-run each environment loss through a gradient closure to obtain
   `∇_ψ R_e`. When `hirm.detach_features=true`, the representation output is
   detached inside the closure so the alignment penalty does not backpropagate
   into φ.
3. Optionally L2-normalise gradients before alignment.
4. Compute either cosine misalignment (`1 - mean cos`) or gradient variance.
5. Combine average risk with `λ_align * penalty`.

During training we freeze φ via `model.freeze_phi` when requested, which simply
marks all non-ψ parameters as non-trainable. Alignment diagnostics are streamed
to `reports/artifacts/<timestamp>_<exp>/seeds/<seed>/train/alignment_head.csv` via
`infra.io.write_alignment_csv` for downstream analysis.
