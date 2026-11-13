# Reproduce the coordinate-decoding benchmark

Run the commands below from the repository root after [installing the package](../README.md) and [obtaining the GraphML inputs](DATA.md). Only the two primary connectome collections are required for this workflow. For a quick check without data, use `python scripts/check_release.py`.

## From graphs to predictions

```sh
python scripts/run_analysis.py --stage inputs
python scripts/run_analysis.py --stage prepare
python scripts/run_analysis.py --stage main --workers 2
python scripts/run_analysis.py --stage template --workers 2
python scripts/run_analysis.py --stage sensitivity --workers 2
python scripts/run_analysis.py --stage summarize
python scripts/run_analysis.py --stage audit
python scripts/plot_results.py
```

The input check matches graphs to the ordered hashes in [the study design](../configs/design.json). Preparation computes the participant responses and templates using only the fitting participants for each split. The next three stages fit the individual-graph models, template controls, and scaling comparison. Summarization reconstructs validation selections and paired intervals. The audit checks saved predictions, numerical solutions, and the effect of projecting predictions back inside the allowed bounds. The plotting command reads the resulting tables, not the published CSVs.

Set `QSF_WORKDIR` before running to choose where derived data and models are written (default: `work/`). The primary response tensors alone require about 2.6 GB, excluding templates, fitted models, and participant-level predictions. Start with one or two workers. A completed stage or cell is reused only if its artifact checks pass. Keep a partial or failed run for diagnosis; use a fresh work directory for a clean restart. Generated outputs do not replace the aggregate tables in `results/`.

## Compare representations

After the primary preparation stage, run the following to compare QSF with channel ablations and shortest-path features, and to withhold target-region coordinates across all fitting and validation participants:

```sh
python scripts/run_controls.py --stage check
python scripts/run_controls.py --stage prepare
python scripts/run_controls.py --stage fit --workers 2
python scripts/run_controls.py --stage summarize
```

The 1,260 configurations share the primary partitions, target masks, and random references. The channel ablations change the dynamic features but retain the probability-weighted source-coordinate moments. In the unseen-region task, the target vertices and atlas correspondence are still known; only their coordinate supervision is withheld.

Two OASIS template fits in the unseen-region task encountered SVD nonconvergence. In that case the runner solves the same positive-ridge objective by QR reduction and augmented least squares, records the recovery, and checks the solution independently. The penalty grid and selection criterion do not change, and no participant is discarded.

For spacing, selected decoder radii, the second HCP scale, or clinical visit linkage, continue to the [anatomical and clinical workflows](ANATOMY.md).

## Design and verification

- [Primary protocol](../configs/protocol.txt) and [representation-control protocol](../configs/controls_protocol.txt): model grids, evaluation rules, and selections. Five overlapping participant holdouts are not family-disjoint HCP folds or an external confirmation sample.
- [Study design](../configs/design.json): input order, splits, target masks, and training-derived source rankings. To recompute rankings, run `python scripts/verify_rankings.py --cohort hcp83` and `python scripts/verify_rankings.py --cohort oasis`. A discrepancy, including one involving nearly tied scores, requires investigation; the commands do not replace the rankings.
- [Primary checks](../results/release_validation.json): the study audited 2,682 primary cells and 1,887,752 participant-family prediction records. The portability checks cover numerical functions, preparation examples, and selected saved predictions; they do not constitute a second fit of all primary models.
- [Control checks](../results/controls_verification.json): all 1,260 control fits reproduced their saved arrays and predictions in the recorded environment. The checks also cover 84 aggregate rows, 96 paired contrasts, independent numerical solutions, withheld-coordinate perturbations, and save/resume behavior. Reproducing the same inputs is not independent scientific replication.

The public command-line entry points are in `scripts/`. The benchmark, data loading, scoring, and controls are in `src/qsf_sfh/`; `src/brain_quantum/` supplies numerical helpers. Local row keys preserve ordering for paired comparisons, but do not make participant-level predictions suitable for public release.
