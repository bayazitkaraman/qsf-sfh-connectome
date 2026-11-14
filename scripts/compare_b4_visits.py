"""Compare the B4 export and the clinical analysis input; report aggregate counts only."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import re

import clinical_visits as analysis


def read_b4(path):
    visits = defaultdict(list)
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            subject = row["OASISID"].strip()
            token = re.fullmatch(r"(OAS3\d+)_UDSb4_d(-?\d+)", row["OASIS_session_label"].strip())
            if not token or token[1] != subject or int(token[2]) != float(row["days_to_visit"]):
                raise ValueError("B4 participant or session day is inconsistent.")
            day = int(token[2])
            if (subject, day) in seen:
                raise ValueError("Duplicate B4 participant and visit day.")
            seen.add((subject, day))
            text = row["CDRTOT"].strip()
            cdr = None if text.casefold() in analysis.MISSING else float(text)
            if cdr not in (None, 0, 0.5, 1, 2, 3):
                raise ValueError("Unexpected global CDR.")
            dx = row["dx1"].strip()
            dx = None if dx.casefold() in analysis.MISSING else dx
            if cdr is not None or dx is not None:
                visits[subject].append((day, cdr, dx, row["dx1_code"].strip()))
    return visits, len(seen)


def indexed(visits):
    return {(subject, day): (cdr, dx, None if code.casefold() in analysis.MISSING else code)
            for subject, records in visits.items() for day, cdr, dx, code in records}


def compare(b4, prepared, cohort):
    left, n = read_b4(b4)
    right, m, _ = analysis.read_visits(prepared)
    a, b = indexed(left), indexed(right)
    shared = a.keys() & b.keys()
    differences = {"visit_keys": len(a.keys() ^ b.keys())}
    for i, field in enumerate(("global_cdr", "primary_diagnosis", "primary_diagnosis_code")):
        differences[field] = sum(a[k][i] != b[k][i] for k in shared)
    windows = []
    for days in analysis.WINDOWS:
        x, y = (analysis.link(cohort, v, days) for v in (left, right))
        diff = {}
        for field, name in (("matched", "matched_status"), ("lag", "selected_visit_day"),
                            ("cdr", "cdr_group"), ("dx", "diagnosis_group"),
                            ("no_code", "diagnosis_code_missing")):
            # The scan day is fixed for each participant; equal lags imply equal visit days.
            equal = x[field].eq(y[field]) | (x[field].isna() & y[field].isna())
            diff[name] = int((~equal).sum())
        windows.append(dict(days=days, participants=len(x), b4_matched=int(x.matched.sum()),
                            prepared_matched=int(y.matched.sum()),
                            b4_held_out_matched=int((x.held & x.matched).sum()),
                            prepared_held_out_matched=int((y.held & y.matched).sum()),
                            differences=diff))
    return dict(inputs=["OASIS-3 Form B4 visit export", "Merged longitudinal analysis table"],
                b4_rows=n, prepared_rows=m, b4_usable_visits=len(a), prepared_usable_visits=len(b),
                usable_visit_differences=differences, windows=windows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b4_csv", type=Path)
    parser.add_argument("prepared_csv", type=Path)
    args = parser.parse_args()
    cohort, _ = analysis.load_cohort()
    report = compare(args.b4_csv, args.prepared_csv, cohort)
    print(json.dumps(report, indent=2, allow_nan=False))
    if any(report["usable_visit_differences"].values()) or any(
            any(w["differences"].values()) for w in report["windows"]):
        raise SystemExit(1)
