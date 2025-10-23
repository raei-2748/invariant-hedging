# Paper reproduction playbook

This guide provides the exact commands, environment assumptions, and artefact
checkpoints required to regenerate the paper figures and tables from scratch.
It mirrors the configuration used to build the "paper" snapshot referenced in
the manuscript.

## Reference environment

| Component | Reference value |
| --- | --- |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10.13 |
| Hardware | 8 vCPU (Intel Ice Lake), 32 GB RAM, **no GPU required** |
| Expected wall-clock | ≈5 minutes end-to-end |

The commands below were validated on the configuration above. Slower laptops
can expect proportionally longer runtimes, but the workflow remains entirely
CPU-compatible.

## Step-by-step commands

1. **Stage the dataset**
   ```bash
   make data
   ```
   Copies `data/spy_sample.csv` into the staged cache under `data/raw/` and
   refreshes checksums for provenance. Replace the source file with your
   institutional SPY export before running this step for the full reproduction.
   *(Expected runtime: <1 s).* 

2. **Dry-run the orchestration script**
   ```bash
   tools/run_of_record.sh --dry-run
   ```
   Prints the training and evaluation commands that will run, verifying that
   Hydra configs resolve correctly on your machine. *(Expected runtime: <1 s).* 

3. **Train and evaluate the paper configuration**
   ```bash
   make paper
   ```
   Executes `configs/train/paper.yaml` and `configs/eval/paper.yaml`, writing
   artefacts to `reports/paper_runs/` and `reports/paper_eval/`. Checkpoints,
   resolved configs, and diagnostics CSV files are created here. *(Expected runtime: ≈3
   min on the reference CPU).* 

4. **Generate publication-ready tables and plots**
   ```bash
   make report-paper
   ```
   Aggregates the evaluation outputs into `reports/paper/` using
   `configs/report/paper.yaml`. Produces LaTeX/CSV tables, scorecard heatmaps,
   and a provenance manifest. *(Expected runtime: ≈2 min on the reference CPU).* 

### Optional follow-up

- `make report` — rebuilds the full 30-seed aggregate using
  `configs/report/default.yaml`. This is unnecessary for the single-seed smoke
  check but reproduces the full paper appendix when all seeds are available.

## Provenance and artifact tracking

- **Git state.** Record the commit hash from `reports/paper_runs/*/metadata.json`
  and `reports/paper_eval/*/metadata.json`. The script captures a `git_status_clean`
  flag; rerun after committing local patches so the flag is `true`.
- **Data lineage.** Retain the original SPY export path and acquisition
  agreement. When sharing artefacts, redact raw prices and provide aggregate
  metrics only.
- **Runtime manifests.** `reports/paper/manifests/aggregate_manifest.json`
  contains hashes for every diagnostics CSV included in the report. Archive this
  manifest with the final paper submission.
- **Hardware notes.** Include CPU model, RAM, and whether a GPU was available in
  any reproduction log. The official snapshot is CPU-only, but documenting
  accelerators helps others interpret runtime differences.

By following the sequence above and storing the generated manifests, another
researcher can audit the pipeline and regenerate the same tables and figures
without additional assumptions.
