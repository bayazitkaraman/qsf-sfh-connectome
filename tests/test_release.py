"""Portable input matching and the released cell-to-summary workflow."""
from pathlib import Path
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from brain_quantum.analysis_suite import compute_walk_stack
from brain_quantum.dense_heat_capacity_control import dense_heat_kernels
from brain_quantum.qrc_connectome import normalized_laplacian
from qsf_sfh import benchmark as ex
from qsf_sfh.data import load_design, locate_graphs
from qsf_sfh.summarize import audit_cell, paired


class ReleaseTests(unittest.TestCase):
    def test_input_matching_uses_content_and_recorded_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'a.graphml').write_bytes(b'first')
            (root/'b.graphml').write_bytes(b'second')
            (root/'copy.graphml').write_bytes(b'first')
            expected = [hashlib.sha256(b'second').hexdigest(), hashlib.sha256(b'first').hexdigest()]
            design = {'cohorts': {'hcp83': {'graph_sha256': expected}}}
            with patch('qsf_sfh.data.load_design', return_value=design):
                actual = locate_graphs('hcp83', root)
            self.assertEqual([p.read_bytes() for p in actual], [b'second', b'first'])

    def test_missing_input_refuses_reduced_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            design = {'cohorts': {'hcp83': {'graph_sha256': ['0'*64]}}}
            with patch('qsf_sfh.data.load_design', return_value=design):
                with self.assertRaisesRegex(ValueError, 'No reduced-cohort'):
                    locate_graphs('hcp83', temp)

    def test_released_design_keeps_calibration_separate(self):
        design = load_design()
        for cohort, n in [('hcp83',1062),('oasis',695)]:
            for parts in design['cohorts'][cohort]['partitions'].values():
                self.assertEqual(sorted(sum([parts[p] for p in ('fit','validation','calibration','test')],[])), list(range(n)))
                self.assertFalse(set(parts['calibration']) & set(parts['train']))

    def test_synthetic_cells_replay_saved_predictions(self):
        rng = np.random.default_rng(20260919)
        weights, quantum, heat = [], [], []
        for _ in range(10):
            w = rng.uniform(.1,2,(8,8)).astype(np.float32)
            w = (w+w.T)/2
            np.fill_diagonal(w,0)
            weights.append(w)
            quantum.append(ex.qsf_stack(compute_walk_stack(normalized_laplacian(w),[.25,.5,1,2,4,8])))
            heat.append(dense_heat_kernels(SimpleNamespace(weights=w),np.geomspace(.25,8,18).tolist()))
        positions = rng.normal(size=(10,8,3))
        positions[8,0] = np.nan
        positions[9,6] = np.nan
        data = dict(qsf=np.stack(quantum),heat=np.stack(heat),positions=positions,
                    extent=np.full(10,10.),weights=np.stack(weights),
                    subjects=[f'row{i:04d}' for i in range(10)])
        parts = dict(fit=[0,1,2,3,4],validation=[5,6],train=list(range(7)),calibration=[7],test=[8,9])
        template_weights = np.asarray(weights[:5],float).mean(axis=0)
        data['templates'] = {1:dict(weights=template_weights,fit_ids=parts['fit'],
            qsf=ex.qsf_stack(compute_walk_stack(normalized_laplacian(template_weights),[.25,.5,1,2,4,8])),
            heat=dense_heat_kernels(SimpleNamespace(weights=template_weights),np.geomspace(.25,8,18).tolist()))}
        with tempfile.TemporaryDirectory() as temp, patch.object(ex,'OUT',Path(temp)), patch.object(ex,'DATA',data):
            for encoder in ex.ENCODERS:
                for graph, scaling in [('individual','tanh3'),('template','tanh3'),('individual','group_floor')]:
                    spec = dict(cohort='synthetic',split=1,mask=1,count=3,policy='random',encoder=encoder,
                                graph=graph,scaling=scaling,sources=[0,1,2],targets=[3,4,5,6,7],partitions=parts)
                    ex.run_cell(spec)
                    folder = ex.directory(spec)
                    saved, choices, metrics = audit_cell(folder)
                    self.assertEqual(saved,spec)
                    self.assertEqual(len(metrics),8)
                    self.assertIn(choices['robust']['family'],('projection','bounded_residual'))
                    before = (folder/'COMPLETE.json').read_bytes()
                    ex.run_cell(spec)
                    self.assertEqual(before,(folder/'COMPLETE.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
