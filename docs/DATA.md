# Data Access and Eligibility

Obtain the derived structural connectomes through the
[BrainGraph download pages](https://braingraph.org/cms/download-pit-group-connectomes/).
Observe the BrainGraph, HCP, and OASIS source-data conditions. This repository
does not redistribute GraphML data or grant permission beyond those terms.

- HCP: `repeated_10_scale_33.7z`, the nominal 86-node release with 83 actual graph labels, Lausanne scale33, one million streamlines with ten repeated reconstructions.
- OASIS-3: `oasis3_graphmls_scale1.7z`, Lausanne2018 scale1, 124 labels, Connectome Mapper 3.1 probabilistic tractography.

The HCP analysis retains 1062 records after two release-provenance exclusions:
one withdrawn diffusion dataset and one derivative for which a withdrawn run
could not be ruled out as an input. The OASIS analysis uses the first available
session per participant, with one incomplete graph excluded, retaining 695.
These are derivative/provenance checks, not independently performed raw-image QC.
No participant is removed because of model prediction error.

`configs/design.json` gives the SHA-256 of every included GraphML file in the
original analysis order. It contains no raw filenames, participant identifiers,
coordinates, or clinical metadata. Memberships reference positions in that
ordered hash list. The input loader searches your data directory recursively,
matches the expected hashes, and refuses to run with a missing or changed input.
Extra graphs are ignored; a different release is not silently substituted.
Duplicates with identical bytes represent the same input, not additional people.

Download archives must be extracted before matching. A convenient layout is:

```text
data/hcp83/*.graphml
data/oasis/*.graphml
```

An existing directory can be used instead, for example in PowerShell:

```powershell
$env:QSF_HCP83_DATA = 'D:\connectomes\hcp'
$env:QSF_OASIS_DATA = 'D:\connectomes\oasis'
python scripts/run_analysis.py --stage inputs
```

Do not rename or edit GraphML contents to force a match. If the source download
changes or becomes unavailable, a complete reproduction requires the matching
source release from its provider. The code does not bypass source access terms.

Graph-label correspondence is retained. Only protected targets with finite
coordinates are scored. Fourteen OASIS hippocampal labels never have finite
coordinates; their vertices remain in the graph but are not positioned sources
or scored targets. The source budget counts selected labels, not necessarily
finite source coordinates for every participant.

The full graph, correspondence, and permitted source coordinates are available
at prediction time. Test-target coordinates enter scoring and post-prediction
audits, not feature construction, source selection, output bounds, or alignment.
