# Release Process

This project uses lightweight release candidates to capture the exact assets referenced in the accompanying paper. Follow the checklist below whenever preparing a new tagged release (for example `v0.9-paper-rc`).

## 1. Prep the repository

1. Ensure the working tree is clean and that all required changes have landed on the default branch.
2. Update `configs/paper/` with the final configuration files that reproduce the paper results.
3. Refresh `meta/paper_provenance.json` so it records the dataset snapshot, training commit, and any caveats for the release.
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
   - The updated `meta/paper_provenance.json` file.
   - Representative metrics such as `outputs/_phase1_snapshot/final_metrics.json` and `outputs/_phase1_snapshot/metrics.jsonl`.
4. Inspect the ZIP contents to ensure no extraneous files or sensitive data slipped in.

## 4. Tag and publish

1. Create an annotated tag following the naming convention `v<MAJOR>.<MINOR>-paper-rc` (for example `v0.9-paper-rc`).
2. Push the tag to GitHub. The `Prepare Paper Release` workflow will run automatically and upload the release artifact for archival.
3. Draft release notes summarizing the paper updates and link to the packaged artifact.
4. Verify the final artifact uploaded by the workflow matches the dry-run output and attach it to the GitHub Release if desired.

## 5. Manual verification checklist

- [ ] Artifact ZIP exists and unzips cleanly.
- [ ] Configuration files correspond to the release tag and pass a quick sanity check.
- [ ] Metrics match the values reported in the paper tables.
- [ ] `meta/paper_provenance.json` is filled in and contains no placeholders.
- [ ] Release notes reference any supplemental assets stored under `releases/<tag>/`.

Following this process ensures that each tagged release captures the reproducibility assets required for the paper while maintaining a clear audit trail.
