from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from score_anatomy import nearest_other
from clinical_visits import link
from qsf_sfh.data import ROOT, load_design
import json


class AnatomyTests(unittest.TestCase):
    def test_spacing_ignores_self_and_missing(self):
        p=np.array([[0.,0.,0.],[3.,4.,0.],[3.,0.,0.],[np.nan]*3])
        np.testing.assert_array_equal(nearest_other(p)[:3],[3.,4.,3.])

    def test_window_uses_one_visit_and_earlier_tie(self):
        cohort=[('row0000','synthetic',100,110,True)]
        visits={'synthetic':[(10,0,'Cognitively normal','1'),(190,1,'AD Dementia','2')]}
        row=link(cohort,visits,90).iloc[0]
        self.assertTrue(row.tie and row.matched)
        self.assertEqual((row.lag,row.cdr,row.dx),(-90,'0','Cognitively normal'))
        self.assertFalse(link(cohort,visits,89).iloc[0].matched)

    def test_hcp129_hash_count_matches_frozen_cohort(self):
        hashes=json.loads((ROOT/'configs/hcp129_inputs.json').read_text())['graph_sha256']
        self.assertEqual(len(set(hashes)),len(load_design()['cohorts']['hcp83']['graph_sha256']))
        self.assertEqual(len(hashes),1062)
