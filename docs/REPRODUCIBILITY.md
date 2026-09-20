# Reproducing the Consolidation

The numerical protocol was frozen on September 18, 2026. Its original byte-exact
text is `configs/protocol.txt`; the recorded SHA-256 is
`f7c88cd027a005e7dd2d9372adee9ebc596df02912523398dc25e0f4b8829bc6`.
The protocol describes the completed experiment, not an untouched preregistration.

## Code and Inputs

`src/qsf_sfh/model.py` and `core.py` retain the original numerical algorithms;
their import paths are adapted for a package. Modules under `src/brain_quantum`
are reused dependencies, not separate recommended analysis pipelines.
`benchmark.py` adapts file locations and raw-data preparation from the frozen
runner. The design file retains all partitions, target masks, and source ranks
by ordered input index. No subject-level data are supplied with the release.

The loader gives local records stable `row0000`-style keys in the original order.
The original participant IDs were already sorted in that order. This preserves
paired alignment and bootstrap sampling order while avoiding identifiers in
the configuration. These local keys are not anonymization guarantees for local
prediction files; treat all record-level outputs as restricted working data.

The release manifest checks source/configuration bytes. The results manifest
checks shipped aggregate CSV bytes. Git line-ending conversion is disabled so
these checks behave consistently across operating systems. Editing tracked
scientific code intentionally invalidates the release check; changes require
a new versioned manifest, not silently bypassing the check.

## Execution

1. Run `scripts/check_release.py` and the unit tests.
2. Obtain and verify the exact GraphML inputs with `--stage inputs`.
3. Run `--stage prepare` to calculate individual responses and fitting-only templates.
4. Run `--stage main`, then `--stage template`, then `--stage sensitivity`.
5. Run `--stage summarize` to reconstruct metrics, replay selections, and compute paired intervals.
6. Run `--stage audit` for scored-target projection strata and independent solver checks.
7. Run `scripts/plot_results.py` for numerical figures from the regenerated outputs.

The main experiment uses fixed training-derived source rankings from the original
run. To independently recompute those rankings from the same inputs:

```sh
python scripts/verify_rankings.py --cohort hcp83
python scripts/verify_rankings.py --cohort oasis
```

This checks the saved rankings rather than replacing them. Small cross-platform
floating-point differences can affect nearly tied selection scores; a mismatch
must be investigated, not silently overwritten. The model remains an exploratory
evaluation using five overlapping holdouts, not family-disjoint HCP folds.

## Resources and Resume

Response tensors alone occupy roughly 2.6 GB (2.4 GiB) in 32-bit floating point, in addition
to graph data, templates, fitted models, and record-level predictions. Allow ample
disk space and begin with one or two worker processes; each fit also allocates
design matrices. Full fitting is substantially more expensive than the tests.

Set `QSF_WORKDIR` before invoking the scripts to choose a local output directory.
Completed data preparations and cells are reused only after their artifact hashes
are checked. Partial cells are rejected rather than guessed complete. Preserve
a failed directory for diagnosis and rerun in a new work directory when necessary.
The code does not delete earlier runs or overwrite shipped aggregate results.

## Verification Scope

The original completed analysis audited 2682 cells and 1,887,752 repeated
participant-family prediction records. Those are historical full-analysis checks,
not a claim that release preparation repeated all fitting. The repository tests
exercise features, source-only information access, bounds, solver agreement,
selection rules, and portable input/output behavior.

Packaging checks compare the released functions and saved design against the
original implementation and outputs. They do not establish independent scientific
replication. The exact tests performed for this release are reported in
`results/release_validation.json`; no new full-cohort refit is claimed there.

`results/` includes only approved aggregate summaries. Fresh record-level arrays,
input-index mappings, source graphs, and local paths must not be committed.
