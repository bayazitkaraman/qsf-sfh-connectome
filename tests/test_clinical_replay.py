import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_clinical_replay as replay
import compare_b4_visits as reconciliation


class ClinicalReplayTests(unittest.TestCase):
    def test_reconciliation_distinguishes_raw_cdr_from_its_group(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            b4, prepared = root / "b4.csv", root / "prepared.csv"
            # Deliberately synthetic participant key, outside the study identifier range.
            key = "OAS3" + "0" * 9
            b4.write_text("OASISID,OASIS_session_label,days_to_visit,CDRTOT,dx1,dx1_code\n"
                          f"{key},{key}_UDSb4_d-001,-1,2,AD Dementia,\n")
            prepared.write_text("subject_id,session_id,days_to_visit,CDRTOT,dx1,dx1_code\n"
                                f"{key},d-001,-1,1,AD Dementia,3\n")
            report = reconciliation.compare(b4, prepared, [("row", key, 99, 110, True)])
            self.assertEqual(report["usable_visit_differences"],
                             dict(visit_keys=0, global_cdr=1, primary_diagnosis=0,
                                  primary_diagnosis_code=1))
            w180, w90 = report["windows"]
            self.assertEqual(w180["differences"]["cdr_group"], 0)
            self.assertEqual(w180["differences"]["diagnosis_code_missing"], 1)
            self.assertEqual(w180["b4_matched"], 1)
            self.assertEqual(w90["b4_matched"], 0)
            self.assertTrue(all(x == 0 for x in w90["differences"].values()))
            self.assertNotIn(key, json.dumps(report))

    def test_check_writes_json_to_new_output_directory(self):
        analysis = replay.analysis
        keys = [f"synthetic{i}" for i in range(20)]
        cohort = [(key, key, 0, 110, True) for key in keys]
        visits = {key: [(0, 0.5 if i < 10 else 0,
                         "AD Dementia" if i < 10 else "Cognitively normal", "1")]
                  for i, key in enumerate(keys)}
        errors = pd.DataFrame({(32, "random_qsf"): [0.06]*10 + [0.04]*10}, index=keys)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.csv"
            source.write_text("synthetic\n")
            archive = root / "results/anatomy"
            archive.mkdir(parents=True)
            def write_summary(_, destination):
                (destination / "groups.csv").write_text("n\n20\n")
            write_summary(source, archive)
            (archive / "clinical_reference.json").write_text(json.dumps({
                "clinical_sha256": analysis.sha(source),
                "outputs_sha256": {"groups": analysis.sha(archive / "groups.csv")},
            }))
            out = root / "new/reports"
            with (patch.object(analysis, "ROOT", root),
                  patch.object(analysis, "run", side_effect=write_summary),
                  patch.object(analysis, "load_cohort", return_value=(cohort, set(keys))),
                  patch.object(analysis, "read_visits", return_value=(visits, 20, 20)),
                  patch.object(analysis, "load_errors", return_value=(errors, 0.0)),
                  redirect_stdout(io.StringIO()) as stdout):
                replay.check(source, out)
            report = json.loads((out / "clinical_replay.json").read_text())
            self.assertEqual(report, json.loads(stdout.getvalue()))
            self.assertEqual(report["source_sha256"], analysis.sha(source))
            self.assertEqual(report["replayed_summaries"], ["groups"])
            self.assertEqual([r["window"] for r in report["diagnosis_code_present_sensitivity"]], [180, 90])
            for row in report["diagnosis_code_present_sensitivity"]:
                self.assertEqual(row["matched"], 20)
                self.assertAlmostEqual(row["estimate"], 0.02)
