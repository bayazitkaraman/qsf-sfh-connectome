"""Synthetic pre-evaluation checks, including end-to-end hidden-label poisoning."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from qsf_sfh import controls as c
import numpy as np
from brain_quantum.dense_heat_capacity_control import dense_heat_kernels
from brain_quantum.source_coordinate_v2 import anchor_features
from qsf_sfh.model import make_bundle


class ControlsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(92026)
        n, nodes = 12, 10
        q, h, p, a, d, weights = [], [], [], [], [], []
        for i in range(n):
            w = rng.uniform(.1,1,(nodes,nodes))
            w = (w+w.T)/2
            np.fill_diagonal(w,0)
            if i == 0:
                w[-1,:] = 0
                w[:,-1] = 0
            lap = c.normalized_laplacian(w)
            walk = c.compute_walk_stack(lap,[.25,.5,1,2,4,8])
            q.append(c.original.qsf_stack(walk))
            h.append(dense_heat_kernels(SimpleNamespace(weights=w),np.geomspace(.25,8,18).tolist()))
            p.append(c.compute_walk_stack(lap,np.geomspace(.25,8,18).tolist())['q_prob'])
            dense = c.compute_walk_stack(lap,np.geomspace(.25,8,9).tolist())
            a.append([v for pair in zip(dense['q_real'],dense['q_imag']) for v in pair])
            d.append(c.all_pairs_shortest_path(w))
            weights.append(w)
        positions = rng.normal(size=(n,nodes,3))
        positions[1,8] = np.nan
        self.data = {k:np.asarray(v,dtype=np.float32) for k,v in dict(qsf=q,heat=h,prob18=p,amp18=a,shortest_path=d).items()}
        self.data.update(positions=positions,extent=np.array([c.position_extent(v) for v in positions]),
                         weights=np.array(weights),subjects=[str(i) for i in range(n)])
        self.parts = dict(fit=list(range(6)),validation=[6,7],train=list(range(8)),test=[8,9,10],calibration=[11])
        self.data['templates'] = {1:dict(qsf=self.data['qsf'][1],heat=self.data['heat'][1],fit_ids=self.parts['fit'])}
        self.spec = dict(cohort='toy',split=1,mask=1,count=2,task='known',encoder='qsf',sources=[0,1],targets=[5,6,7])

    def feature(self, encoder):
        return c.features(self.data,0,np.array([0,1]),np.array([5,6,7]),encoder,self.data['positions'][0,[0,1]],1)

    def test_original_bundle_parity(self):
        for encoder in ('qsf','heat_raw18','heat_norm18'):
            spec = dict(self.spec,encoder=encoder)
            pos, extent, targets = c.learning_view(self.data,spec)
            actual = c.bundle(self.data,spec,targets,self.parts['fit'],pos,extent)
            expected = make_bundle(self.data,np.array(spec['sources']),targets,encoder,self.parts['fit'])
            for attr in ('x','y','finite','center','scale','extent','fallback'):
                np.testing.assert_array_equal(getattr(actual,attr),getattr(expected,attr))

    def test_feature_dimensions(self):
        dims = dict(qsf=42,heat_raw18=42,heat_norm18=42,prob6=18,amp12=30,prob18=42,amp18=42,shortest_path=10,moments=6)
        for encoder, count in dims.items():
            self.assertEqual(self.feature(encoder)[0].shape,(3,count))

    def test_ablation_moments_and_response_are_identical(self):
        full,response = self.feature('qsf')
        for encoder in ('prob6','amp12','prob18','amp18','moments'):
            x,r = self.feature(encoder)
            np.testing.assert_array_equal(x[:,-6:],full[:,-6:])
            np.testing.assert_array_equal(r,response)

    def test_exact_channel_slicing(self):
        full = self.feature('qsf')[0][:,:-6].reshape(3,18,2)
        np.testing.assert_array_equal(self.feature('prob6')[0][:,:-6],full[:,2::3].reshape(3,12))
        np.testing.assert_array_equal(self.feature('amp12')[0][:,:-6],full[:,np.arange(18)%3!=2].reshape(3,24))

    def test_shortest_path_matches_existing_definition(self):
        record = SimpleNamespace(shortest_paths=self.data['shortest_path'][0],
                                 connectome=SimpleNamespace(positions=self.data['positions'][0]))
        x,r = anchor_features(record,np.array([0,1]),'shortest_path')
        new,response = self.feature('shortest_path')
        np.testing.assert_array_equal(new,x[[5,6,7]])
        np.testing.assert_array_equal(response,r[[5,6,7]])

    def test_disconnected_path_convention_and_edgeless_case(self):
        w = np.array([[0.,2.,0.],[2.,0.,0.],[0.,0.,0.]])
        paths = c.all_pairs_shortest_path(w)
        self.assertAlmostEqual(paths[0,2],1.25/(2+1e-9))
        np.testing.assert_array_equal(c.all_pairs_shortest_path(np.zeros((3,3))),0)
        np.testing.assert_array_equal(self.data['prob18'][0,:,0,9],0)
        np.testing.assert_array_equal(self.data['amp18'][0,:,0,9],0)

    def test_unseen_learning_view_omits_hidden_coordinates(self):
        spec = dict(self.spec,task='unseen')
        pos, extent, targets = c.learning_view(self.data,spec)
        self.assertFalse(np.isfinite(pos[:,spec['targets']]).any())
        self.assertFalse(set(targets)&set(spec['sources']+spec['targets']))
        altered = dict(self.data,extent=np.full(12,1e15),positions=self.data['positions'].copy())
        altered['positions'][:,spec['targets']] = -1e15
        p2,e2,t2 = c.learning_view(altered,spec)
        np.testing.assert_array_equal(pos,p2)
        np.testing.assert_array_equal(extent,e2)
        np.testing.assert_array_equal(targets,t2)

    def test_full_hidden_coordinate_poisoning_all_unseen_models(self):
        for encoder in c.UNSEEN:
            spec = dict(self.spec,task='unseen',encoder=encoder)
            result = c.fit_predict(self.data,spec,self.parts)
            self.assertTrue(c.poison_check(self.data,spec,self.parts,result))
            self.assertLess(c.independent_check(result,self.parts),1e-7)

    def test_no_test_coordinate_or_calibration_fitting(self):
        result = c.fit_predict(self.data,self.spec,self.parts)
        changed = dict(self.data,positions=self.data['positions'].copy())
        for i in self.parts['test']:
            changed['positions'][i,self.spec['targets']] += 1e9
        changed['positions'][self.parts['calibration']] = np.nan
        other = c.fit_predict(changed,self.spec,self.parts)
        self.assertEqual(result['grid'],other['grid'])
        np.testing.assert_array_equal(result['prediction'],other['prediction'])

    def test_template_has_no_individual_graph_features(self):
        spec = dict(self.spec,encoder='template_qsf',task='unseen')
        first = c.fit_predict(self.data,spec,self.parts)
        changed = dict(self.data,qsf=np.zeros_like(self.data['qsf']))
        second = c.fit_predict(changed,spec,self.parts)
        np.testing.assert_array_equal(first['prediction'],second['prediction'])

    def test_all_known_methods_select_bounded_models(self):
        for encoder in c.KNOWN:
            result = c.fit_predict(self.data,dict(self.spec,encoder=encoder),self.parts)
            self.assertIn(result['selected']['family'],('projection','bounded_residual'))
            self.assertGreater(result['selected']['alpha'],0)
            b = result['evaluation']
            dist = np.linalg.norm(result['prediction']-b.center[self.parts['test']],axis=2)
            self.assertTrue((dist <= result['selected']['radius']*b.scale[self.parts['test'],None]+1e-10).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
