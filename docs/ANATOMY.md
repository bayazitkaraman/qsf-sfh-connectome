# Anatomical and clinical comparisons

To read the results, open the [aggregate tables](../results/anatomy/). To compute
them, choose a workflow below. All commands run from the repository root and
write under `QSF_WORKDIR` (default: `work/`). Keep participant-level outputs private.

| Question | Script | Supplement tables |
|---|---|---|
| Error relative to nearest other finite centroid; selected radii and outside fractions | `scripts/score_anatomy.py` | S14, S15 |
| Same participants at Lausanne2008 scale60 | `scripts/run_parcellation.py` | S16 |
| Nearest clinical visit and coordinate availability or held-out error | `scripts/clinical_visits.py` | S17 |

## Anatomical spacing and bounds

After [fitting the primary models and controls](REPRODUCIBILITY.md) for both
cohorts, including the template models, score the saved predictions:

```sh
python scripts/score_anatomy.py practical
python scripts/score_anatomy.py bounds
```

The scorer verifies the saved artifacts, the common sources/targets where
applicable, and the published benchmark errors. Participant summaries are
computed before averaging or bootstrapping across people. Bootstrap draws
use the fixed seeds in the scripts. No target coordinate enters fitting here.

## Parcellation scale in HCP

This workflow fits its own models. Supply the HCP robust-connectome
129-node GraphML files for the same participants:

```sh
python scripts/run_parcellation.py --data /path/to/hcp129 --workers 1
```

The ordered 1062 hashes in `configs/hcp129_inputs.json` select exactly the
eligible graphs, in HCP-83 design order. The five partitions come from
`configs/design.json`; the three target masks and nested source rankings
are generated at 129 labels. Preparation needs several GB of writable disk;
the 135 fits are not a quick installation test.

## Clinical visit matching in OASIS

Supply the hash-matched OASIS GraphMLs using their original
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

The first command matches the nearest clinical visit within each window
(180 and 90 days, inclusive). Read [coverage](../results/anatomy/coverage.csv)
for the matched and unmatched counts, [groups](../results/anatomy/groups.csv)
for coordinate availability, and [contrasts](../results/anatomy/contrasts_32.csv)
for the error differences at 32 sources. Availability uses all matched
participants in each group; prediction error uses only those with held-out
predictions. Visits and diagnoses are not inputs to the decoder.
The second checks the study input fingerprint, aggregate replay and sensitivity
to missing numeric diagnosis codes; it is not authentication of an official
export. A different export must be reconciled at the visit level, not declared
equivalent because totals happen to match. Raw visit records, participant
identities, graphs, and predictions are not distributed.
