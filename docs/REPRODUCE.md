# Paper reproduction playbook

This guide documents the exact environment, commands, and runtime expectations required to regenerate every figure and table in *“Robust Generalization for Hedging under Crisis Regime Shifts.”* It is the authoritative reference for the camera-ready v1.0.0 release.

## Reference environment

| Component | Reference value |
| --- | --- |
| OS | Ubuntu 22.04 LTS |
| Python | 3.11.9 (CPython) |
| Hardware | 8 vCPU (Intel Ice Lake), 32 GB RAM, no GPU required |
| Optional GPU | NVIDIA A10 (compute capability 8.6) for accelerated training |
| Apple Silicon | M2 Pro (12c CPU / 19c GPU, 32 GB unified memory) running macOS 14.6 |
| Key libraries | torch 2.3.1, numpy 1.26.4, pandas 2.2.3, matplotlib 3.9.2, scikit-learn 1.5.2, tqdm 4.66.4 |

All dependencies are pinned in [`requirements.txt`](../requirements.txt), [`environment.yml`](../environment.yml), and [`requirements-lock.txt`](../requirements-lock.txt). Use the lock file for exact reproducibility:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
pip install -e .[dev]
```

## Dataset staging

1. **Snapshot data inputs**
   ```bash
   make data
   ```
   Downloads the deterministic SPY sample (`data/spy_sample.csv`) and refreshes checksums. For the full paper reproduction replace the CSV before running the target.

2. **Record provenance**
   ```bash
   tools/run_of_record.sh --dry-run
   ```
   Confirms Hydra resolves all configs, prints the train/eval commands, and writes the provenance header that will later populate `archive/paper_provenance.json`.

## Training & evaluation workflow

1. **Train and evaluate the canonical configuration**
   ```bash
   make paper
   ```
Runs `configs/train/paper.yaml` followed by `configs/eval/paper.yaml`, producing artefacts in `reports/paper_runs/` and `reports/paper_eval/`. Set `SMOKE=1` for a 3-seed sanity pass (~3 min CPU) or omit for the full 30-seed sweep (~45 min CPU, ~12 min with A10 GPU).

2. **Assemble publication assets**
   ```bash
   make report-paper
   ```
   Aggregates diagnostics into `reports/paper/`, generating CSV/LaTeX tables, figure manifests, and a refreshed `archive/paper_provenance.json` record. Expected runtime: ~8 min on the reference CPU.

3. **Crisis robustness evaluation**
   ```bash
   make eval-crisis
   ```
Executes `configs/eval/robustness.yaml`, exporting crisis-shift diagnostics to `reports/paper_eval/robustness/` (~15 min CPU, ~6 min GPU). Seeds are fixed by `seed_list.txt`.

### Apple Silicon notes

- Install the Metal-enabled PyTorch wheel (bundled with the CPU whl) in your venv:
  ```bash
  python3 -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
  ```
- `make paper` now prints the resolved device plus CUDA/MPS availability. When `torch.backends.mps.is_available()` returns `True`,
  the launcher automatically adds `runtime.device=mps runtime.mixed_precision=false` so the full 30-seed sweep runs on the GPU.
- AMP is disabled on MPS for numerical stability. Expect roughly 2–3× faster wall-clock time compared to CPU-only runs on an
  M2 Pro/Max, though certain operators may still fall back to CPU when no Metal kernel exists.
- If the runtime warns that it selected CPU on macOS, verify the PyTorch install and rerun `make data && make paper` once the
  Metal backend is available.

## Figure & table generation

After `make report-paper`, the repository contains a complete set of diagnostics. The following commands materialise the assets cited in the paper:

- **Fig. 2 (Invariant gap vs. worst-group risk)**
  ```bash
  make plot-ig-wg
  ```
  Uses `reports/paper/` tables to recreate the invariance scatter. Output PNG/PDF files live in `reports/figures/ig_vs_wg/`. Runtime: < 1 min, ≈2 GB RAM.

- **Fig. 3 (Capital efficiency frontier)**
  ```bash
  python -m src.visualization.plot_capital_frontier --run_dir "$(ls -td reports/paper/* | head -1)" --out_dir reports/figures/capital_frontier
  ```
  Produces the capital efficiency curve reported in §6. Runtime: ≈2 min CPU, 3 GB RAM.

- **Table 3 (Crisis robustness under regime shifts)**
  ```bash
  make eval-crisis
  ```
  Generates `reports/paper_eval/robustness/final_metrics.json` containing the Crisis IG/WG deltas. Runtime: ≈15 min CPU.

- **Appendix tables (stress diagnostics)**
  ```bash
  python tools/scripts/aggregate.py --config configs/report/default.yaml
  ```
  Rebuilds the full appendix, writing tables under `reports/paper/appendix/`. Runtime: ≈10 min CPU; requires completed full-seed runs.

All scripts respect deterministic seeds embedded in the configs (`seed_list.txt`). To override, set `SEED_OVERRIDE` before invoking `make paper` or edit the Hydra configs directly.

## Summary checklist

| Figure/Table | Command | Notes |
|---------------|----------|-------|
| Fig. 2 (IG vs WG) | `make plot-ig-wg` | Requires `make report-paper`; ~1 min; 2 GB RAM |
| Fig. 3 (Capital frontier) | `python -m src.visualization.plot_capital_frontier --run_dir "$(ls -td reports/paper/* | head -1)" --out_dir reports/figures/capital_frontier` | Ensure `make report-paper`; ~2 min |
| Table 3 (Crisis robustness) | `make eval-crisis` | ≈15 min CPU; produces JSON summary |
| Appendix A (Stress diagnostics) | `python tools/scripts/aggregate.py --config configs/report/default.yaml` | Requires full seed sweep |

## Expected outputs & validation

- `archive/paper_provenance.json` — Git SHA, Hydra configs, platform metadata, and SHA256 fingerprints for every generated asset.
- `reports/paper/manifests/aggregate_manifest.json` — Manifest of figure/table hashes used in the camera-ready bundle.
- `reports/coverage/index.html` — Pytest coverage report generated via `make coverage`.
- `reports/paper/figures/` — Final PNG/PDF assets for inclusion in the manuscript.

Verify success with:

```bash
make coverage
pytest --maxfail=1 --disable-warnings -q
```

Both commands should complete without errors and report the deterministic seed list captured in `seed_list.txt`. Archive the resulting coverage report, manifests, and provenance JSON alongside the submission package.
