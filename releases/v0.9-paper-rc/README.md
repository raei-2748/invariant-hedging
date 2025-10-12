# v0.9 Paper Release Candidate

This directory seeds the structure expected for the `v0.9-paper-rc` artifact bundle. Populate it with the final files before tagging the release:

- Copy the curated configuration files into `configs/paper/`.
- Update `paper_provenance.json` at the repository root with the correct metadata.
- Export the final evaluation metrics (e.g., `outputs/_phase1_snapshot/final_metrics.json`).
- Include any additional documentation or tables required by the paper and link them here.

Once populated, run the `Prepare Paper Release` workflow in dry-run mode to confirm the package layout, then tag the release.
