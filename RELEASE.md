# Release Process

This project uses lightweight release candidates to capture the exact assets referenced in the accompanying paper. Follow the checklist below whenever preparing a new tagged release. `v0.9-paper-rc` serves as the feature freeze for the manuscript; promote the tag to `v1.0-paper` once CI and nightly smoke tests are green on the release candidate.

## 1. Prep the repository

1. Ensure the working tree is clean and that all required changes have landed on the default branch.
2. Update `configs/paper/` with the final configuration files that reproduce the paper results.
3. Refresh `paper_provenance.json` so it records the dataset snapshot, training commit, and any caveats for the release.
4. Export the representative evaluation metrics (for example `outputs/_phase1_snapshot/final_metrics.json`) so that they reflect the tagged commit.
5. Review `releases/<tag>/` for any release-specific notes or supplemental artifacts that should accompany the bundle.

## 2. Required checks

Before tagging a release, the following checks must pass:

- CI: `ci.yml` and `ci.yaml` GitHub Actions workflows should be green on the target commit.
- Linting/tests: run `make test` locally if practical to spot regressions.
- Workflow packaging: run the `Prepare Paper Release` workflow (described below) in dry-run mode to confirm the artifact layout.

## 3. Package verification (dry run)

1. Navigate to **Actions → Prepare Paper Release** in GitHub.
2. Use **Run workflow** and supply:
   - `dry_run`: keep as `true` to avoid publishing anything.
   - `release_tag`: (optional) set to the intended tag such as `v0.9-paper-rc` so the artifact name matches expectations.
3. Download the resulting artifact and confirm it contains:
   - The full `configs/paper/` directory.
   - The updated `paper_provenance.json` file.
   - Representative metrics such as `outputs/_phase1_snapshot/final_metrics.json` and `outputs/_phase1_snapshot/metrics.jsonl`.
4. Inspect the ZIP contents to ensure no extraneous files or sensitive data slipped in.

## 4. Collect release assets

Before tagging the release, assemble the artefacts that will accompany the GitHub Release so downstream users can reproduce the paper snapshot on a clean machine:

1. Regenerate the paper report (or the lite variant) so the rendered tables exist under `outputs/report_paper/`:
   ```bash
   make report-paper
   ```
   For a smaller bundle you can build the lite snapshot instead:
   ```bash
   make report-lite
   ```
2. Update `paper_provenance.json` with the dataset snapshot identifier and the exact training commit hash.
3. Stage the bundle with the packaging helper (writes to `releases/<tag>/` by default):
   ```bash
   python scripts/package_release.py \
       --tag v0.9-paper-rc \
       --report-dir outputs/report_paper \
       --golden scorecard.csv \
       --golden seed_values.csv \
       --docker-digest ghcr.io/acme/invariant-hedging@sha256:YOUR_DIGEST
   ```
   The helper copies the following into the bundle and emits `manifest.json` with checksums:
   - `environment/environment.yml`
   - `environment/docker_digest.txt`
   - `data/data-mini.tar.gz` and `data/data-mini.sha256`
   - `reports/` (tables, figures, manifests)
   - `golden/` CSVs (fast access to the headline tables)
   - `provenance/paper_provenance.json` and `provenance/final_metrics.json` (if available)
4. Inspect the manifest to verify every asset is present:
   ```bash
   cat releases/v0.9-paper-rc/manifest.json | jq '.'
   ```

## 5. Tag and publish

1. Create an annotated tag following the naming convention `v<MAJOR>.<MINOR>-paper-rc` (for example `v0.9-paper-rc`).
2. Push the tag to GitHub. The `Prepare Paper Release` workflow will run automatically and upload the release artifact for archival.
3. Draft release notes summarizing the paper updates and link to the packaged bundle.
4. Verify the final artifact uploaded by the workflow matches the dry-run output and attach the bundle contents (or archive) to the GitHub Release.

## 6. Manual verification checklist

- [ ] Artifact ZIP exists and unzips cleanly.
- [ ] Configuration files correspond to the release tag and pass a quick sanity check.
- [ ] Metrics match the values reported in the paper tables.
- [ ] `paper_provenance.json` is filled in and contains no placeholders.
- [ ] Release notes reference any supplemental assets stored under `releases/<tag>/`.

## 7. Reproduce the paper-lite report from release assets

Verify the bundle supports clean-room reproduction before publishing the final tag. On a fresh machine with Docker installed:

1. Download the staged `releases/<tag>/` directory (or the zipped archive attached to the draft release).
2. Extract `data/data-mini.tar.gz` so the miniature dataset mirrors the repository layout:
   ```bash
   tar -xzf releases/v0.9-paper-rc/data/data-mini.tar.gz -C data/
   ```
3. Launch the recorded container image and rebuild the lite report:
   ```bash
   docker run --rm -it \
     -v "$PWD":/workspace \
     -w /workspace \
     ghcr.io/acme/invariant-hedging@sha256:YOUR_DIGEST \
     bash -lc "pip install -r requirements.txt && make report-lite"
   ```
4. Confirm the regenerated CSVs in `outputs/report_paper/tables/` match the golden copies shipped in the release bundle.

The release is ready once the clean-room run reproduces the lite report without manual tweaks.

Following this process ensures that each tagged release captures the reproducibility assets required for the paper while maintaining a clear audit trail.
