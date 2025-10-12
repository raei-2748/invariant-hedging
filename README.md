# HIRM

[![CI](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml/badge.svg)](https://github.com/raei-2748/invariant-hedging/actions/workflows/ci.yml)
[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10--3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper reproduction guide](https://img.shields.io/badge/reproduce-paper-blue.svg)](REPRODUCE.md)

HIRM (Hedging with IRM) is a research codebase for reproducing the paper results on
robust hedging under regime shifts. This repository now defaults to the
**paper reproduction workflow**: acquiring the SPY dataset snapshot, running the
compact training/evaluation pipeline, and generating the camera-ready tables and
figures.

For a full walkthrough of every command, expected artefacts, and provenance
requirements see [REPRODUCE.md](REPRODUCE.md).

## Quickstart

1. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```
2. **Stage the SPY dataset snapshot**
   ```bash
   make data
   ```
3. **Preview the end-to-end workflow** (prints the commands that will run)
   ```bash
   scripts/run_of_record.sh --dry-run
   ```
4. **Train + evaluate the paper configuration on CPU**
   ```bash
   make paper
   ```
5. **Build publication tables/figures from the paper run**
   ```bash
   make report-paper
   ```
6. *(Optional)* **Rebuild the full report assets for multi-seed runs**
   ```bash
   make report
   ```

The commands above execute in minutes on a single CPU-only workstation; see the
[reproduction playbook](REPRODUCE.md) for the precise runtime profile and
hardware that were used for the reference paper snapshot.

## Data acquisition summary

The experiments rely on SPY option market snapshots bucketed into volatility
regimes. A 5,000-row smoke subset is included as `data/spy_sample.csv` to enable
local testing and CI. Running `make data` copies this file into
`outputs/paper_data/` so the paper configs can locate it without mutating the
raw download.

For the full paper reproduction you must supply the institutional SPY dataset
cited in the paper (2017–2022 daily close-to-close options). Place the CSV (or
parquet) export in `data/` and update `configs/data/real_spy.yaml` with the
filename if it differs from the default. Provenance expectations are documented
in [REPRODUCE.md](REPRODUCE.md#provenance-and-artifact-tracking).

## Paper pipelines

### `make paper`
Runs `scripts/run_of_record.sh`, which trains the compact IRM head model on the
paper configuration (`configs/train/paper.yaml`) and evaluates the resulting
checkpoint with the matching evaluation profile (`configs/eval/paper.yaml`).
Outputs are written to `runs/paper/` (training) and `runs/paper_eval/`
(evaluation), including `final_metrics.json`, per-environment diagnostics, and
Hydra config mirrors.

### `make report`
Generates the full multi-seed aggregation described in the paper using
`configs/report/default.yaml`. This target expects 30-seed runs under `runs/*`
(as produced by the large-scale sweeps) and renders the complete scorecard,
heatmaps, QQ plots, and optional I–R–E 3D projections into
`outputs/report_assets/`. The command is heavier and is not required for the
single-seed smoke reproduction.

### `make report-paper`
Aggregates the latest paper run into camera-ready assets using
`configs/report/paper.yaml`. The output directory `outputs/report_paper/`
contains the scorecard CSV/LaTeX tables, heatmaps, and provenance manifest for
inclusion in the paper appendix.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Missing data/spy_sample.csv` | Dataset not staged | Download or symlink the SPY snapshot into `data/` and re-run `make data`. |
| `Could not locate the latest training run` | `make paper` was interrupted before finishing | Remove partial directories under `runs/paper/` and re-run `make paper`. |
| `No checkpoints saved in run directory` | Training failed before writing `checkpoints/` | Check `runs/paper/*/metrics.jsonl` for stack traces and re-run after addressing the error. |
| `aggregate.py` exits with "No seed files" | Evaluation artefacts missing | Ensure `make paper` completed successfully and `runs/paper_eval/` contains `diagnostics_seed_*.csv`. |

If an issue persists, capture the failing command output and open a discussion
in the repository.

## FAQ

**Is the repository open source?**  Yes. The code is released under the MIT
License, allowing research and commercial use with attribution. See
[LICENSE](LICENSE) for the exact terms.

**What if I get a `ValueError` about CSV columns when running `make report`?**
This indicates a malformed or truncated diagnostics CSV—usually because the
underlying dataset export was interrupted. Re-run `make paper` (or the
multi-seed sweep) after verifying that your SPY dataset is complete and matches
the schema defined in `configs/data/real_spy.yaml`.

**Can I redistribute the SPY dataset?**  No. The SPY market data is licensed
from a commercial provider and cannot be redistributed. The repository only
ships the 5,000-row smoke subset for testing; you must obtain the full dataset
under your own agreement.

## Additional references

- [REPRODUCE.md](REPRODUCE.md): command-by-command reproduction checklist with
  runtime, hardware, and provenance notes.
- [`experiments/`](experiments/): original research plans, baselines, and phase
  summaries for historical context.
- [`scripts/run_of_record.sh`](scripts/run_of_record.sh): orchestration script
  used by `make paper`.
