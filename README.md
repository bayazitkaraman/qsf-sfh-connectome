# Decoding regional coordinates from structural connectomes

Given a brain connectivity graph and the locations of a few reference regions, how well can a model predict the positions of other regions? Measured centroids provide a known answer. The purpose is to compare the anatomical information available to a decoder from different graph representations, not to replace a centroid that can already be measured from an atlas.

**Quantum-Walk-Inspired Source Fields (QSF)** describe a region by its real, imaginary, and probability responses to each reference across graph time scales. A bounded ridge decoder maps those responses and source-coordinate features to a three-dimensional position. **Source-Field Hierarchy (SFH)** is one reference-selection policy, not a second decoder. Both are classical numerical methods.

## Explore the results

The study evaluates 1,062 HCP participants (83 regions) and 695 OASIS-3 participants (124 regions) separately. Each model receives the complete weighted graph, regional correspondence, and the available coordinates of 8, 16, or 32 reference regions. The coordinates of the targets being predicted are withheld.

At **32 common random references**, the mean normalized root-mean-square error (nRMSE; lower is better) on the known-region task was:

| Representation | HCP-83 | OASIS-3 |
| --- | ---: | ---: |
| QSF on the participant's graph | 0.10455 | 0.04983 |
| Raw heat on the participant's graph | 0.11906 | 0.07108 |
| Normalized heat on the participant's graph | 0.12022 | 0.07351 |
| QSF on a training-only template graph | 0.02976 | 0.02883 |
| Source-aligned atlas | 0.02289 | 0.02297 |

Each held-out participant has equal weight. On this task, the same target labels can supply training coordinates in other participants. QSF gave lower error than both heat encodings, but the aligned atlas was more accurate than any graph representation shown above. When those labels supplied **no training or validation coordinates**, QSF still beat both heat encodings at 32 sources, with higher errors of 0.18822 (HCP-83) and 0.12110 (OASIS-3). The labels and graphs remain known in this test.

| Question | Open the aggregate results |
| --- | --- |
| How do QSF, heat, a template graph, and an aligned atlas compare? | [Errors by budget and reference policy](results/track_summary.csv) |
| What happens without a response channel, with shortest paths, or without target-label coordinate supervision? | [Representation controls](results/controls_summary.csv) |
| How often does a prediction fall within the distance to the target's nearest other centroid? Does choosing the references help? | [Anatomical spacing](results/anatomy/practical.csv) and [paired differences](results/anatomy/practical_paired.csv) |
| How restrictive are the decoder's bounds? | [Radii and fraction of targets outside](results/anatomy/bounds.csv) |
| Does the QSF advantage persist in the same HCP participants at 129 regions? | [Parcellation comparison](results/anatomy/hcp129.csv) |
| Which OASIS participants have a nearby clinical visit, and how does error vary by group? | [Matching coverage](results/anatomy/coverage.csv) and [descriptive contrasts](results/anatomy/contrasts_32.csv) |

The cohorts informed method development. The 129-region analysis tests a different scale in the same HCP participants, not an independent population. The clinical comparisons describe subsets with a matched visit; clinical labels do not enter model fitting.

## Try it without participant data

Clone the repository and enter it:

```sh
git clone https://github.com/bayazitkaraman/qsf-sfh-connectome.git
cd qsf-sfh-connectome
python -m venv .venv
```

Use Python 3.12 (the study used 3.12.14). Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/check_release.py
python scripts/plot_results.py --input results
```

The first script verifies the distributed files; the second writes ten figure panels as PNG and PDF to `work/figures/` by default. Neither command downloads data or fits a model. To run the numerical and input-handling tests, use `python -m unittest discover -s tests -v`.

## Reproduce the fits

1. **[Obtain the inputs](docs/DATA.md).** Match the BrainGraph derivatives by file hash. Access to OASIS clinical visits is only needed for the clinical comparison.
2. **[Run the primary benchmark and representation controls](docs/REPRODUCIBILITY.md).** This guide covers preparation, fitting, scoring, resource requirements, and verification.
3. **[Run the anatomical and clinical analyses](docs/ANATOMY.md).** Choose a workflow for anatomical spacing, decoder bounds, the 129-region HCP parcellation, or clinical visit matching.

The [study snapshot (v1.2.0)](https://github.com/bayazitkaraman/qsf-sfh-connectome/tree/v1.2.0) identifies the exact code and aggregate results cited in *Anatomical Information in Source-Conditioned Connectome Responses: Bounded Coordinate Decoding and Reference Placement*. Raw graphs, clinical visit records, and participant-level predictions are not distributed.

**Author:** Bayazit Karaman, Department of Computer Science, Florida Polytechnic University. **Contact:** bkaraman@floridapoly.edu. The [MIT license](LICENSE) covers the software; the source datasets retain their own access and use conditions.
