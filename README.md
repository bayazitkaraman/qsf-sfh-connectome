# QSF and SFH Connectome Analysis

Code and aggregate results for **Anatomical Information in Source-Conditioned
Connectome Responses: Bounded Coordinate Decoding and Reference Placement**.

Quantum-Walk-Inspired Source Fields (QSF) describe each brain region through
multiscale responses to known reference regions. A shared decoder predicts
protected target coordinates. Source-Field Hierarchy (SFH) is one of the fixed
reference-selection policies evaluated alongside random, coverage, spectral,
and response-discrimination policies. These are classical numerical methods,
not quantum-computing or biological quantum-process claims.

## Current Analysis

Version 1.0.0 packages the September 18, 2026 consolidation: 1062 HCP-derived
83-node connectomes and 695 OASIS-3-derived 124-node connectomes, evaluated
separately with five overlapping participant holdouts, three protected target
masks, and budgets of 8, 16, and 32 source labels. All models receive the complete
weighted graph, atlas correspondence, source identities, and available source
coordinates; protected test-target coordinates are withheld from prediction.

At 32 common random sources, equal-participant mean normalized RMSE was:

| Representation | HCP-83 | OASIS |
| --- | ---: | ---: |
| Individual-graph QSF | 0.10455 | 0.04983 |
| Raw heat, matched feature count | 0.11906 | 0.07108 |
| Normalized heat, matched feature count | 0.12022 | 0.07351 |
| Fitting-only template QSF | 0.02976 | 0.02883 |
| Source-aligned atlas | 0.02289 | 0.02297 |

Nonspatial reference selection reduced individual-graph QSF means to 0.10007 and
0.04622 on the same targets. QSF improved on the two tested heat representations,
but template graphs and the aligned atlas were more accurate. These exploratory
results do not demonstrate a predictive benefit from participant-specific
connectivity. Both cohorts had already informed development; the holdouts are
not an untouched external replication.

## Repository Layout

```text
configs/     Hash-identified inputs, fixed partitions, source rankings, protocol
docs/        Data access and reproduction details
results/     Current aggregate CSV results and their checksums
scripts/     Analysis, source-ranking, plotting, and verification entry points
src/         Consolidated model and reused numerical helpers
tests/       Synthetic and numerical regression tests
```

There are no manuscript files, raw connectomes, participant-level prediction
records, private review notes, IRB documents, or model caches in the current tree.
This repository starts with the consolidated implementation. Internal helper
names are preserved where they identify code used in the audited analysis; the current entry point
is `scripts/run_analysis.py`, not the older helper modules' command-line programs.

## Setup

The reported environment used Python 3.12.14. From a cloned checkout:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on Linux/macOS, then:

```sh
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/check_release.py
python -m unittest discover -s tests -v
```

The checks and tests do not require the datasets. Run from the repository root;
the release uses its adjacent `configs/` and `results/` directories and is
intended for a checkout with an editable installation, not a standalone wheel.

## Data and Reproduction

Obtain the original BrainGraph HCP and OASIS derivatives under their source-data
terms. See [data access](docs/DATA.md) for the exact input matching and eligibility
conventions. Put extracted GraphML files under `data/hcp83/` and `data/oasis/`,
or set `QSF_HCP83_DATA` and `QSF_OASIS_DATA` to their existing locations.

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

Preparation regenerates responses from verified GraphML files; it does not
require the author's private pickle caches. Full fitting is computationally
substantial and is not a quick installation test. `--stage all` executes the
full sequence. See [reproduction details](docs/REPRODUCIBILITY.md) for resource
requirements, resume behavior, validation scope, and optional ranking checks.

New participant-level outputs stay under ignored `work/` directories, or an
external directory set by `QSF_WORKDIR`. They never overwrite the shipped
aggregate results. The existing `results/*.csv` are the audited study outputs,
not numbers recomputed by a fresh full-cohort run of this release package.

To plot the shipped aggregates without downloading data, use
`python scripts/plot_results.py --input results`. This produces ten panels;
the two record-level error distributions require outputs from a completed run.

## Interpretation and License

The method evaluates coordinate information in source-conditioned graph
representations. It is not an atlas-free localizer, clinical prediction system,
or claim that selected references are physiological hubs. Output bounds limit
extrapolation but can worsen error for targets outside their permitted region.

Software is covered by the MIT terms in [LICENSE](LICENSE). Original HCP,
OASIS, and BrainGraph data retain their own access and use conditions.

Author: Bayazit Karaman, Department of Computer Science, Florida Polytechnic
University. Contact: bkaraman@floridapoly.edu.
