"""Synthetic checks before consolidation outcomes are generated."""
import sys
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace
from brain_quantum.analysis_suite import compute_walk_stack
from brain_quantum.decoder_robustness import RidgePath
from brain_quantum.dense_heat_capacity_control import dense_heat_kernels
from brain_quantum.qrc_connectome import normalized_laplacian
from brain_quantum.source_coordinate_v2 import anchor_features
from qsf_sfh.core import project
from qsf_sfh.model import (FloorPath, features, grouped_scale, make_bundle, policy_choice,
                   projection_diagnostics, robust_choice)
from qsf_sfh.benchmark import qsf_stack


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(19081)
        w = self.rng.uniform(.1,1,(8,8))
        w = (w+w.T)/2
        np.fill_diagonal(w,0)
        self.weights = w
        self.walk = compute_walk_stack(normalized_laplacian(w),[.25,.5,1,2,4,8])
        self.q = qsf_stack(self.walk)
        self.h = dense_heat_kernels(SimpleNamespace(weights=w),np.geomspace(.25,8,18).tolist())
        self.pos = self.rng.normal(size=(8,3))
        self.sources = np.array([0,1,2])
        self.targets = np.array([3,4,5,6,7])

    def test_qsf_replays_existing_feature_layout(self):
        record = SimpleNamespace(connectome=SimpleNamespace(positions=self.pos),
            q_real=[v.astype(np.float32) for v in self.walk['q_real']],
            q_imag=[v.astype(np.float32) for v in self.walk['q_imag']],
            q_prob=[v.astype(np.float32) for v in self.walk['q_prob']])
        old,response = anchor_features(record,self.sources,'qrc')
        new,r = features(self.q,self.pos[self.sources],self.sources,self.targets,'qsf')
        np.testing.assert_array_equal(new,old[self.targets])
        np.testing.assert_array_equal(r,response[self.targets])

    def test_matched_dimensions_and_full_row_normalization(self):
        for encoder,stack in [('qsf',self.q),('heat_raw18',self.h),('heat_norm18',self.h)]:
            x,_ = features(stack,self.pos[self.sources],self.sources,self.targets,encoder)
            self.assertEqual(x.shape,(5,60))
        x,_ = features(self.h,self.pos[self.sources],self.sources,self.targets,'heat_norm18')
        normalized = self.h/self.h.sum(axis=2,keepdims=True)
        np.testing.assert_allclose(x[:,:54],normalized[:,self.targets][:,:,self.sources].transpose(1,0,2).reshape(5,54))

    def test_source_only_feature_and_frame_invariance(self):
        data = dict(qsf=np.stack([self.q]*4),heat=np.stack([self.h]*4),
                    positions=np.stack([self.pos]*4),extent=np.ones(4))
        before = make_bundle(data,self.sources,self.targets,'qsf',[0,1])
        data['positions'][3,self.targets] += 1e6
        after = make_bundle(data,self.sources,self.targets,'qsf',[0,1])
        np.testing.assert_array_equal(before.x,after.x)
        np.testing.assert_array_equal(before.center,after.center)
        np.testing.assert_array_equal(before.scale,after.scale)

    def test_template_ignores_individual_propagators(self):
        data = dict(qsf=np.stack([self.q]*4),heat=np.stack([self.h]*4),
                    positions=np.stack([self.pos]*4),extent=np.ones(4))
        template = dict(qsf=self.q.copy(),heat=self.h.copy())
        before = make_bundle(data,self.sources,self.targets,'qsf',[0,1],template)
        data['qsf'][3] = self.rng.normal(size=self.q.shape)
        after = make_bundle(data,self.sources,self.targets,'qsf',[0,1],template)
        np.testing.assert_array_equal(before.x,after.x)

    def test_isolate_cross_component_zeros(self):
        w = self.weights.copy()
        w[0,:] = 0
        w[:,0] = 0
        stack = qsf_stack(compute_walk_stack(normalized_laplacian(w),[.25,.5,1,2,4,8]))
        np.testing.assert_array_equal(stack[:,1:,0],0)

    def test_group_floor_is_positive_and_local(self):
        x = np.zeros((20,42))
        x[:,1] = np.arange(20)
        scale = grouped_scale(x,2)
        self.assertGreater(scale[0,0],0)
        np.testing.assert_allclose(scale[0,2:4],.1)
        with self.assertRaises(ValueError):
            grouped_scale(x,3)

    def test_floor_solver_matches_augmented_least_squares(self):
        x = self.rng.normal(size=(90,42))
        x[:,0] *= 1e-12
        y = self.rng.normal(size=(90,6))
        p = FloorPath(x,y,2)
        m = p.model(.1)
        z = (x-m.mean)/m.scale-m.transformed_mean
        a = np.vstack([z,np.sqrt(.1)*np.eye(42)])
        b = np.vstack([y-m.y_mean,np.zeros((42,6))])
        coef = np.linalg.lstsq(a,b,rcond=None)[0]
        np.testing.assert_allclose(coef,m.coef,rtol=1e-9,atol=1e-10)

    def test_primary_inputs_bounded_for_extreme_values(self):
        p = RidgePath(self.rng.normal(size=(90,42)),self.rng.normal(size=(90,6)),'tanh3')
        self.assertTrue(np.isfinite(p.model(.01).predict(np.full((3,42),1e200))).all())

    def test_projection_facts_and_idempotence(self):
        center = np.zeros((3,8,3))
        truth = self.rng.normal(size=center.shape)
        raw = self.rng.normal(size=center.shape)*30
        scale = np.ones(3)
        pred,_ = project(raw,center,scale[:,None],2.)
        projection_diagnostics(raw,pred,truth,center,scale,2.,np.ones((3,8),bool),np.ones(3))
        np.testing.assert_allclose(project(pred,center,scale[:,None],2.)[0],pred)

    def test_robust_selection_never_unbounded(self):
        rows = {'projection':dict(family='projection',validation_nrmse=1.,validation_nmse=1.),
                'bounded_residual':dict(family='bounded_residual',validation_nrmse=.995,validation_nmse=.995)}
        self.assertEqual(robust_choice(rows)['family'],'projection')
        rows['bounded_residual']['validation_nrmse'] = .8
        self.assertEqual(robust_choice(rows)['family'],'bounded_residual')

    def test_policy_selection_validation_safeguard(self):
        rows = [dict(policy='random',validation_nrmse=1.,validation_nmse=1.),
                dict(policy='graph_coverage',validation_nrmse=.8,validation_nmse=1.1)]
        self.assertEqual(policy_choice(rows)['policy'],'random')
        rows[1]['validation_nmse'] = .9
        self.assertEqual(policy_choice(rows)['policy'],'graph_coverage')


if __name__ == '__main__':
    unittest.main()
