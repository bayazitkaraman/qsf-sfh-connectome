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

Start with the [OASIS-3](https://sites.wustl.edu/oasisbrains/home/access/) Form B4
visit export obtained under your own access approval. Run the practical scorer
first and set `QSF_OASIS_DATA` to the hash-matched OASIS GraphMLs with their
original filenames. The six fields below are sufficient for Table S17.

| Official B4 column | Input to `clinical_visits.py` | Meaning |
| --- | --- | --- |
| `OASISID` | `subject_id` | Keep the participant key unchanged. |
| `OASIS_session_label` | `session_id` | Keep the final `d<signed day>` token, including any minus sign and leading zeros. |
| `days_to_visit` | `days_to_visit` | Study day of the **visit**, not the number of days between scan and visit. |
| `CDRTOT` | `CDRTOT` | Global Clinical Dementia Rating in this export; do not substitute the sum of boxes (`CDRSUM`). |
| `dx1` | `dx1` | Recorded primary diagnosis text. |
| `dx1_code` | `dx1_code` | Recorded primary diagnosis code; leave missing codes empty. |

Run this Python snippet locally. Replace the two paths with locations in a
private, access-controlled directory **outside the Git checkout**. Use a new
output filename. Keep all visits, including rows with a missing field.

```python
from pathlib import Path
import pandas as pd

source = Path("/private/path/Form B4 Global Staging CDR Standard and Supplemental.csv")
target = Path("/private/path/b4_visits.csv")
fields = ["OASISID", "OASIS_session_label", "days_to_visit", "CDRTOT", "dx1", "dx1_code"]
visits = pd.read_csv(source, usecols=fields, dtype=str, keep_default_na=False)
keys = visits["OASIS_session_label"].str.extract(r"^(OAS3\d+)_UDSb4_(d-?\d+)$")
if keys.isna().any().any() or not keys[0].eq(visits["OASISID"]).all():
    raise ValueError("Unexpected participant or session label in the B4 export.")
days = pd.to_numeric(visits["days_to_visit"], errors="raise")
if not pd.to_numeric(keys[1].str[1:]).eq(days).all():
    raise ValueError("Session day does not agree with days_to_visit.")
if pd.DataFrame({"id": keys[0], "day": days}).duplicated().any():
    raise ValueError("Duplicate participant and visit day; inspect the export.")
visits = visits.rename(columns={"OASISID": "subject_id"}).assign(session_id=keys[1])
columns = ["subject_id", "session_id", "days_to_visit", "CDRTOT", "dx1", "dx1_code"]
visits.to_csv(target, columns=columns, index=False, mode="x")
```

```sh
python scripts/clinical_visits.py /private/path/b4_visits.csv --out /private/path/b4_results
```

Use a fresh results directory for a new input file. This command selects the
nearest eligible visit within 180 and 90 days (inclusive). Compare the generated
[coverage](../results/anatomy/coverage.csv), [groups](../results/anatomy/groups.csv),
[errors](../results/anatomy/errors.csv), and [contrasts](../results/anatomy/contrasts_32.csv)
with the published aggregates. Coordinate availability uses the full group;
prediction errors use only participants with held-out predictions. Clinical
variables do not enter model fitting. Keep the original and prepared visit CSVs
private. The [source comparison](../results/anatomy/b4_reconciliation.json)
names the two inputs and reports aggregate discrepancies. All 8,625 usable visits
agreed on participant and visit day, global CDR, primary diagnosis, and its
code (including missingness). Across the 695 study participants, both files
selected the same visits and clinical groups at 180 and 90 days. This agreement
does not require the source files to have identical bytes.

With the local primary inputs prepared, the following compares the official
B4 export with the study's merged input and prints only aggregate counts.
The replay command also requires the practical scorer outputs and the exact
merged input identified by its fingerprint. It checks the four published
clinical tables and writes `clinical_replay.json` to the specified directory.

```sh
python scripts/compare_b4_visits.py /private/path/form_b4.csv /private/path/merged_study_input.csv
python scripts/check_clinical_replay.py /private/path/merged_study_input.csv --out /private/path/replay_results
```
