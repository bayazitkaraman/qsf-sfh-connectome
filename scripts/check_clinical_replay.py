"""Reproduce the frozen clinical summaries and test sensitivity to absent diagnosis codes.

The sensitivity changes only which linked records enter the diagnosis contrast.
It neither rematches visits nor refits or excludes anyone from the primary models.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import clinical_visits as analysis


def check(source, output):
    archive = analysis.ROOT / "results" / "anatomy"
    manifest = json.loads((archive / "clinical_reference.json").read_text())
    if analysis.sha(source) != manifest["clinical_sha256"]:
        raise ValueError("Different input file; reconcile visit records before finalizing.")
    with TemporaryDirectory() as folder:
        replay = Path(folder)
        analysis.run(source, replay)
        for name, digest in manifest["outputs_sha256"].items():
            filename = name + ".csv"
            assert analysis.sha(archive / filename) == digest
            pd.testing.assert_frame_equal(pd.read_csv(archive / filename),
                                          pd.read_csv(replay / filename),
                                          check_exact=False, rtol=1e-10, atol=1e-12)

    cohort, test = analysis.load_cohort()
    visits, _, _ = analysis.read_visits(source)
    errors, delta = analysis.load_errors(test)
    primary = errors[32, "random_qsf"]
    results = []
    for window in analysis.WINDOWS:
        linked = analysis.link(cohort, visits, window)
        usable = linked[linked.matched & ~linked.no_code]
        ref = usable.index[usable.held & usable.dx.eq("Cognitively normal")]
        ad = usable.index[usable.held & usable.dx.eq("AD Dementia")]
        assert min(len(ref), len(ad)) >= analysis.MIN_N
        contrast = analysis.interval(primary[ad], primary[ref])
        assert np.isfinite(list(contrast.values())).all()
        results.append(dict(window=window, matched=len(usable),
                            held_out=int(usable.held.sum()), ad_held=len(ad),
                            normal_held=len(ref), **contrast))
    report = dict(source_sha256=analysis.sha(source),
                  endpoint_reconstruction_max_abs=delta,
                  replayed_summaries=list(manifest["outputs_sha256"]),
                  diagnosis_code_present_sensitivity=results)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "clinical_replay.json"
    payload = json.dumps(report, indent=2, allow_nan=False)
    path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clinical_csv", type=Path)
    parser.add_argument("--out",type=Path,default=analysis.WORK/"clinical")
    args=parser.parse_args()
    check(args.clinical_csv,args.out)
