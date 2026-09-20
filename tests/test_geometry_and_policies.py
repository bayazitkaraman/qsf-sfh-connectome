import unittest

import numpy as np

from brain_quantum.analysis_suite import compute_walk_stack
from brain_quantum.decoder_robustness import RidgePath
from brain_quantum.qrc_connectome import normalized_laplacian

from qsf_sfh.core import (Bundle, choose_track, discrimination_sequence, farthest_sequence,
                  fit_families, losses, project, reconstruct, source_frame, spectral_sequence)


class ExtensionTests(unittest.TestCase):
    def test_projection_bound_and_no_mutation(self):
        prediction = np.array([[[3.,4.,0.],[.1,0.,0.]], [[20.,0.,0.],[1.,2.,0.]]])
        original = prediction.copy()
        center = np.zeros_like(prediction)
        projected, clipped = project(prediction, center, np.array([[1.],[2.]]), 2.)
        self.assertTrue(np.all(np.linalg.norm(projected,axis=-1) <= [[2.],[4.]]))
        np.testing.assert_array_equal(clipped, [[True,False],[True,False]])
        np.testing.assert_array_equal(prediction,original)

    def test_source_frame_fallbacks_use_only_sources(self):
        response = np.array([[0.,0.],[1.,3.]])
        center, scale, no_sources, fallback, zero_rows = source_frame(response, [[0.,0.,0.],[4.,0.,0.]], [99.,99.,99.], 8.)
        np.testing.assert_allclose(center, [[2.,0.,0.],[3.,0.,0.]])
        self.assertEqual((scale,no_sources,fallback,zero_rows),(2.,0,0,1))
        center, scale, no_sources, fallback, _ = source_frame(response, [[np.nan]*3]*2,[1.,2.,3.],8.)
        np.testing.assert_allclose(center, [[1.,2.,3.]]*2)
        self.assertEqual((scale,no_sources,fallback),(8.,1,1))

    def test_shared_design_is_separate_ridge_regressions(self):
        rng = np.random.default_rng(19)
        x,y,r = rng.normal(size=(80,12)),rng.normal(size=(80,3)),rng.normal(size=(80,3))
        a = RidgePath(x,np.column_stack([y,r]),"asinh").model(.01)
        b = RidgePath(x,y,"asinh").model(.01)
        c = RidgePath(x,r,"asinh").model(.01)
        np.testing.assert_allclose(a.coef[:,:3],b.coef,atol=1e-12)
        np.testing.assert_allclose(a.coef[:,3:],c.coef,atol=1e-12)

    def test_track_selection_cannot_use_test_results(self):
        base = {"family":"ordinary","policy":"random","validation_nrmse":.1,"validation_nmse":.02}
        weak = {**base,"family":"projection","validation_nrmse":.0995,"test_nrmse":0.}
        risky = {**base,"family":"residual","validation_nrmse":.08,"validation_nmse":.021}
        self.assertEqual(choose_track([weak,risky],base),base)
        good = {**base,"family":"residual","validation_nrmse":.09,"validation_nmse":.015,"test_nrmse":99.}
        self.assertEqual(choose_track([good],base),good)

    def test_discrimination_greedy_matches_direct_objective(self):
        a = np.array([[1.,0.,.2],[.1,4.,.1],[.5,.5,.5]])
        order, objective = discrimination_sequence(a,[7,8,9])
        remaining = [0,1,2]
        total = np.zeros(3)
        expected = []
        for _ in range(3):
            winner=max(remaining,key=lambda i:np.log1p(total+a[i]).sum())
            expected.append([7,8,9][winner])
            total += a[winner]
            remaining.remove(winner)
        self.assertEqual(order,expected)
        self.assertTrue(np.all(np.diff(objective)>=0))

    def test_log_objective_diminishing_returns(self):
        a = np.random.default_rng(23).uniform(size=(4,12))
        gain_empty = np.log1p(a[3]).sum()
        gain_one = (np.log1p(a[0]+a[3])-np.log1p(a[0])).sum()
        gain_two = (np.log1p(a[0]+a[1]+a[3])-np.log1p(a[0]+a[1])).sum()
        self.assertGreaterEqual(gain_empty,gain_one)
        self.assertGreaterEqual(gain_one,gain_two)

    def test_symmetry_and_added_non_target_source(self):
        w=np.diag(np.ones(4),1)+np.diag(np.ones(4),-1)
        walk=compute_walk_stack(normalized_laplacian(w),[.25,.5,1.,2.,4.,8.])
        for channel in ("q_real","q_imag","q_prob","c_heat_raw"):
            center=np.array([a[[0,4],2] for a in walk[channel]])
            np.testing.assert_allclose(center[:,0],center[:,1],atol=1e-14)
        q_added=np.array([a[[0,4],1] for channel in ("q_real","q_imag","q_prob") for a in walk[channel]])
        h_added=np.array([a[[0,4],1] for a in walk["c_heat_raw"]])
        self.assertGreater(np.linalg.norm(q_added[:,0]-q_added[:,1]),.01)
        self.assertGreater(np.linalg.norm(h_added[:,0]-h_added[:,1]),.01)

    def test_shared_prediction_error_bound(self):
        y=np.array([[-1.,0.,0.],[1.,0.,0.]])
        for shared in ([.3,.2,.1],[0.,0.,0.],[10.,20.,30.]):
            mse=np.mean(np.sum((y-shared)**2,axis=1))
            self.assertGreaterEqual(mse,np.sum((y[0]-y[1])**2)/4)

    def test_spectral_sampling_restricted_to_candidate_pool(self):
        w=np.diag(np.ones(5),1)+np.diag(np.ones(5),-1)
        order=spectral_sequence(w,np.array([1,2,3,4]),width=3)
        self.assertEqual(set(order),{1,2,3,4})
        self.assertEqual(len(order),4)
        _,u=np.linalg.eigh(normalized_laplacian(w))
        u=u[:,:3]
        gram=.001*np.eye(3)
        remaining=[1,2,3,4]
        expected=[]
        for _ in range(4):
            current=np.linalg.slogdet(gram)[1]
            values=np.array([np.linalg.slogdet(gram+np.outer(u[i],u[i]))[1]-current for i in remaining])
            tied=np.flatnonzero(np.isclose(values,values.max(),rtol=1e-10,atol=1e-12))
            winner=min(remaining[i] for i in tied)
            expected.append(winner)
            gram+=np.outer(u[winner],u[winner])
            remaining.remove(winner)
        self.assertEqual(order,expected)

    def test_farthest_prefix_has_no_protected_nodes(self):
        distance=np.abs(np.arange(6)[:,None]-np.arange(6)[None,:])
        order=farthest_sequence(distance,np.array([1,2,3,4]),1)
        self.assertEqual(order[:2],[1,4])
        self.assertEqual(set(order),{1,2,3,4})

    def test_synthetic_full_fit_and_bounds(self):
        rng=np.random.default_rng(97)
        x=rng.normal(size=(18,5,7))
        y=x[...,:3]*2+3
        b=Bundle(x,y,np.ones((18,5),bool),np.full(18,10.),np.ones((18,5,3)),np.full(18,2.),np.zeros(18),np.zeros((18,3)))
        grid,chosen,models,_,_=fit_families(b,b,list(range(9)),list(range(9,13)),list(range(13)))
        self.assertEqual(len(grid),192)
        for family,row in chosen.items():
            raw=models[family].predict(x[13:].reshape(-1,7)).reshape(5,5,6)
            p,_=reconstruct(raw,b,list(range(13,18)),family,row["radius"])
            nr,nm=losses(p,b,list(range(13,18)))
            np.testing.assert_allclose(nr**2,nm)
            if row["radius"]:
                self.assertLessEqual(np.max(np.linalg.norm(p-b.center[13:],axis=2)),2*row["radius"]+1e-12)


if __name__ == "__main__":
    unittest.main()
