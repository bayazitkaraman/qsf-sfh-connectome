"""Link a locally supplied clinical visit table to scan days and summarize held-out errors."""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qsf_sfh.data import ROOT, WORK, locate_graphs, check_release

PRIMARY = WORK/'consolidation/oasis'
DATA = PRIMARY/'data'
DESIGN = ROOT/'configs/design.json'
PERSONS = WORK/'anatomy/practical_participant.csv.gz'
WINDOWS = (180, 90)
MODELS = ("random_qsf", "selected_qsf", "random_heat_raw18", "random_heat_norm18")
DX = ("Cognitively normal", "uncertain dementia", "AD Dementia")
MISSING = {"", ".", "na", "n/a", "nan", "unknown", "not available", "not assessed"}
MIN_N = 10
SEED = 20261112


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_cohort():
    release = json.loads(DESIGN.read_text())["cohorts"]["oasis"]
    paths = locate_graphs("oasis")
    manifest = json.loads((DATA / "COMPLETE.json").read_text())["artifacts"]
    for name in ("positions.npy", "subjects.json"):
        assert sha(DATA / name) == manifest[name]
    subjects = json.loads((DATA / "subjects.json").read_text())
    assert subjects == [f'row{i:04d}' for i in range(len(paths))]
    pos = np.load(DATA / "positions.npy", mmap_mode="r")
    assert pos.shape == (695, 124, 3)
    finite = np.isfinite(pos).all(axis=-1)
    assert finite.any(axis=0).sum() == 110 and finite.sum() == 76315
    test = {subjects[i] for fold in release["partitions"].values() for i in fold["test"]}
    assert len(test) == 461
    cohort = []
    for s, path, n in zip(subjects, paths, finite.sum(axis=1), strict=True):
        m = re.search(r"(OAS3\d+)_ses-d(\d+)_", path.name)
        if not m: raise ValueError("Use the original OASIS GraphML filenames with participant and scan day.")
        cohort.append((s, m[1], int(m[2]), int(n), s in test))
    assert len({row[1] for row in cohort}) == 695
    return cohort, test


def read_visits(path):
    visits = defaultdict(list)
    keys = set()
    total = 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            total += 1
            s = r["subject_id"].strip()
            m = re.fullmatch(r"d(-?\d+)", r["session_id"].strip())
            assert re.fullmatch(r"OAS3\d+", s) and m
            day = int(m[1])
            assert float(r["days_to_visit"]) == day and (s, day) not in keys
            keys.add((s, day))
            c = r["CDRTOT"].strip()
            c = None if c.casefold() in MISSING else float(c)
            assert c is None or c in (0, 0.5, 1, 2, 3)
            label = r["dx1"].strip()
            if label.casefold() in MISSING:
                label = None
            code = r["dx1_code"].strip()
            if c is not None or label is not None:
                visits[s].append((day, c, label, code))
    return visits, total, len({s for s, _ in keys})


def link(cohort, visits, window):
    rows = []
    for s, oid, scan, n, held in cohort:
        eligible = sorted((v for v in visits.get(oid, ()) if abs(v[0]-scan) <= window),
                          key=lambda v: (abs(v[0]-scan), v[0]))
        v = eligible[0] if eligible else None
        tie = len(eligible) > 1 and abs(eligible[1][0]-scan) == abs(v[0]-scan)
        c, label = (v[1], v[2]) if v else (None, None)
        rows.append(dict(subject=s, finite=n, held=held, matched=v is not None,
                         lag=(v[0]-scan if v else np.nan), tie=tie,
                         cdr=("Unmatched" if not v else "Missing" if c is None else
                              "0" if c == 0 else "0.5" if c == 0.5 else ">=1"),
                         dx=("Unmatched" if not v else "Missing" if not label else
                             label if label in DX else "Other recorded"),
                         no_code=bool(label and v[3].casefold() in MISSING)))
    result = pd.DataFrame(rows).set_index("subject")
    assert result.index.is_unique
    return result


def bootmeans(x, rng):
    return np.concatenate([x[rng.integers(len(x), size=(100, len(x)))].mean(axis=1)
                           for _ in range(50)])


def interval(x, ref=None):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(SEED)
    sample = bootmeans(x, rng)
    value = x.mean()
    if ref is not None:
        ref = np.asarray(ref, dtype=float)
        sample -= bootmeans(ref, rng)
        value -= ref.mean()
    lo, hi = np.quantile(sample, [.025, .975])
    return dict(estimate=value, lo95=lo, hi95=hi)


def load_errors(test):
    df = pd.read_csv(PERSONS, usecols=["cohort", "task", "count", "model", "subject", "nrmse"])
    df = df[(df.cohort == "oasis") & (df.task == "known") & df.model.isin(MODELS)]
    published = pd.read_csv(WORK / "anatomy/practical.csv").set_index(["cohort","task","count","model"])
    errors = df.pivot(index="subject", columns=["count", "model"], values="nrmse").sort_index()
    assert set(errors.index) == test and errors.shape == (461, 12)
    assert np.isfinite(errors.to_numpy()).all()
    for (k, model), val in errors.mean().items():
        np.testing.assert_allclose(val, published.loc[("oasis","known",k,model),"nrmse"], atol=1e-12, rtol=0)
    # Reconstruct the primary endpoint independently from all 15 frozen test cells.
    records = defaultdict(list)
    for split in range(1, 6):
        for mask in range(1, 4):
            cell = PRIMARY / "cells" / f"s{split}_m{mask}_k32_random_qsf_individual_tanh3"
            man = json.loads((cell/"COMPLETE.json").read_text())["artifacts"]
            for name in ("selections.json", "predictions.npz"):
                assert sha(cell/name) == man[name]
            family = json.loads((cell/"selections.json").read_text())["robust"]["family"]
            ix = ("ordinary", "projection", "residual", "bounded_residual").index(family)
            with np.load(cell/"predictions.npz") as p:
                sq = np.square(p["prediction"][ix] - p["truth"]).sum(axis=2)
                loss = np.sqrt(np.nanmean(sq, axis=1))/p["extent"]
                for s, v in zip(p["subject_ids"], loss, strict=True):
                    records[str(s)].append(v)
    reconstructed = pd.Series({s:np.mean(vals) for s, vals in records.items()}).sort_index()
    assert set(reconstructed.index) == test
    delta = abs(reconstructed - errors[32, "random_qsf"]).max()
    assert delta < 1e-12
    return errors, float(delta)


def run(source, dest):
    input_hash = sha(source)
    if (dest/"checks.json").exists():
        old = json.loads((dest/"checks.json").read_text())
        if old["clinical_sha256"] != input_hash:
            raise ValueError("Use a separate output directory for a different clinical source.")
    cohort, test = load_cohort()
    visits, nrows, npeople = read_visits(source)
    errors, delta = load_errors(test)
    groups, metrics, contrasts, coverage = [], [], [], []
    primary = errors[32, "random_qsf"]
    heat = errors[32, "random_heat_raw18"]
    prev = None
    for w in WINDOWS:
        data = link(cohort, visits, w)
        matched = data[data.matched]
        if prev is not None:
            pd.testing.assert_frame_equal(matched, prev.loc[matched.index])
        prev = matched
        for label, sub in [("all", data), ("held-out", data[data.held])]:
            m = sub[sub.matched]
            coverage.append(dict(window=w, cohort=label, eligible=len(sub), matched=len(m),
                unmatched=len(sub)-len(m), exact_day=(m.lag==0).sum(),
                earlier=(m.lag<0).sum(), later=(m.lag>0).sum(), ties=m.tie.sum(),
                median_abs_days=m.lag.abs().median(), p90_abs_days=m.lag.abs().quantile(.9),
                matched_missing_diagnosis=(m.dx=="Missing").sum(), matched_missing_cdr=(m.cdr=="Missing").sum(),
                recorded_label_without_code=m.no_code.sum()))
        for axis, ref in (("cdr", "0"), ("dx", "Cognitively normal")):
            ref_idx = data.index[(data[axis]==ref) & data.held]
            for label, g in data.groupby(axis, sort=True):
                idx = g.index[g.held]
                n = len(g)
                key = dict(window=w, stratification=axis, group=label)
                groups.append(dict(**key, n=n, pct_cohort=100*n/len(data), held_out=len(idx),
                    mean_finite=g.finite.mean() if n>=MIN_N else None,
                    few_coords=(g.finite<110).sum() if n>=MIN_N else None,
                    pct_fewer_than_110=100*(g.finite<110).mean() if n>=MIN_N else None))
                if len(idx)<MIN_N:
                    continue
                vals = errors.loc[idx]
                for (k, model), v in vals.mean().items():
                    metrics.append(dict(**key, held_out=len(idx), sources=k,model=model,mean_nrmse=v))
                if label not in ("Missing", "Unmatched"):
                    contrasts.append(dict(**key, comparison="QSF minus raw heat, within participants",
                        held_out=len(idx), **interval(primary[idx]-heat[idx])))
                if label not in ("Missing", "Unmatched", ref):
                    contrasts.append(dict(**key, comparison="QSF group minus reference, between participants",
                        held_out=len(idx), **interval(primary[idx], primary[ref_idx])))
        for axis in ["cdr", "dx"]:
            rows = [r for r in groups if r["stratification"]==axis and r["window"]==w]
            assert sum(r["n"] for r in rows)==695 and sum(r["held_out"] for r in rows)==461
    dest.mkdir(parents=True, exist_ok=True)
    files = {"coverage":coverage, "groups":groups, "errors":metrics, "contrasts_32":contrasts}
    for name, rows in files.items():
        pd.DataFrame(rows).to_csv(dest/f"{name}.csv", index=False, float_format="%.10g")
    log = dict(status="Descriptive linkage to the user-supplied clinical table; replay is not independent source validation.",
               clinical_sha256=input_hash, clinical_rows=nrows, clinical_people=npeople,
               design_sha256=sha(DESIGN), participants_file_sha256=sha(PERSONS),
               script_sha256=sha(Path(__file__)),
               primary_max_absolute_reconstruction_difference=delta, seed=SEED,
               outputs_sha256={n:sha(dest/f"{n}.csv") for n in files})
    (dest/"checks.json").write_text(json.dumps(log, indent=2)+"\n", encoding="utf-8")
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clinical_csv", type=Path)
    parser.add_argument("--out", type=Path, default=WORK/"clinical")
    args = parser.parse_args()
    check_release()
    run(args.clinical_csv, args.out)
