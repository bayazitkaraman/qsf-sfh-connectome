# Anatomical and clinical comparisons (version 1.2.0)

The scripts use the same environment and work directory as the primary
analysis. Run commands from the repository root. `results/anatomy/` contains
the archived aggregate tables. Generated files go under `QSF_WORKDIR`
(default: `work/`); participant-level outputs must remain private.

| Question | Script | Published tables |
|---|---|---|
| Error relative to nearest other finite centroid; selected radii and outside fractions | `scripts/score_anatomy.py` | S14, S15 |
| Same participants at Lausanne2008 scale60 | `scripts/run_parcellation.py` | S16 |
| Nearest clinical visit and coordinate availability or held-out error | `scripts/clinical_visits.py` | S17 |

After running the primary and control fit commands in the main README for
both cohorts (including template cells), score the saved predictions:

```sh
python scripts/score_anatomy.py practical
python scripts/score_anatomy.py bounds
```

The scorer verifies the saved artifacts, the common sources/targets where
applicable, and the published benchmark errors. Participant summaries are
computed before averaging or bootstrapping across people. Bootstrap draws
use the fixed seeds in the scripts. No target coordinate enters fitting here.

For the separately obtained HCP robust-connectome 129-node GraphML files:

```sh
python scripts/run_parcellation.py --data /path/to/hcp129 --workers 1
```

The ordered 1062 hashes in `configs/hcp129_inputs.json` select exactly the
eligible graphs, in HCP-83 design order. The five partitions come from
`configs/design.json`; the three target masks and nested source rankings
are generated at 129 labels. Preparation needs several GB of writable disk;
the 135 fits are not a quick installation test.

For OASIS, supply the hash-matched GraphMLs using their original
participant/session filenames via `QSF_OASIS_DATA`. Run the practical
scorer first. Supply a locally authorized longitudinal visit CSV with columns
`subject_id,session_id,days_to_visit,CDRTOT,dx1,dx1_code`.
Session codes are d-prefixed signed study days. Participant and day must
identify a unique visit; formats are validated, not guessed. The study
used a merged longitudinal table, not an imaging-filtered MRI/PET merge.

```sh
python scripts/clinical_visits.py /private/path/visits.csv
python scripts/check_clinical_replay.py /private/path/identical-study-input.csv
```

The first command computes both inclusive windows (180 and 90 days).
The second checks the study input fingerprint, aggregate replay and sensitivity
to missing numeric diagnosis codes; it is not authentication of an official
export. A different export must be reconciled at the visit level, not declared
equivalent because totals happen to match. Raw visit records, participant
identities, graphs, and predictions are not distributed.

`python scripts/check_release.py` verifies distributed checksums without data.
The portable entry points reuse the study scoring and fitting routines.
The released aggregates are the study results, not evidence of a second
independent full refit or independent diagnostic confirmation.
