"""Solver recovery and participant-level reporting invariants."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from brain_quantum.decoder_robustness import RidgePath
from qsf_sfh.control_solver import AugmentedPath, fit
from qsf_sfh.control_summary import contrast, summarize
from qsf_sfh import model


class ControlSolverTests(unittest.TestCase):
    def test_augmented_solution_matches_direct_ridge(self):
        rng = np.random.default_rng(20260920)
        for rows,cols in ((110,24),(50,70),(130,24)):
            x = rng.normal(size=(rows,cols))
            x[:,0] = 0
            x[:,1] = x[:,2]
            x[:,3] *= 1e-12
            y = rng.normal(size=(rows,6))
            direct,other = RidgePath(x,y,'tanh3'),AugmentedPath(x,y,'tanh3',1)
            for alpha in (.001,.01,.1,1.,10.,25.,100.,1000.):
                a,b = direct.model(alpha),other.model(alpha)
                np.testing.assert_allclose(a.coef,b.coef,atol=1e-9,rtol=1e-8)
                np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-9,rtol=1e-8)

    def test_fallback_only_handles_svd_nonconvergence(self):
        sentinel = [None,None,None,{}]
        with patch.object(model,'fit',side_effect=[np.linalg.LinAlgError('SVD did not converge'),sentinel]) as mock:
            actual = fit(None,None,[],[],[],8,'tanh3')
            self.assertIs(actual,sentinel)
            self.assertIs(mock.call_args.kwargs['path_factory'],AugmentedPath)
            self.assertEqual(actual[-1]['solver_recovery'],'QR plus augmented least squares')
        with patch.object(model,'fit',side_effect=np.linalg.LinAlgError('unrelated error')):
            with self.assertRaisesRegex(np.linalg.LinAlgError,'unrelated error'):
                fit(None,None,[],[],[],8,'tanh3')

    def test_positive_penalties_only(self):
        path = AugmentedPath(np.eye(6),np.eye(6),'tanh3',1)
        for alpha in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):
                path.model(alpha)

    def test_repeated_records_are_not_extra_participants(self):
        frame = pd.DataFrame([dict(cohort='toy',task='known',count=8,encoder='qsf',subject_id=s,
              split=i,mask=1,normalized_rmse=x,normalized_mse=x*x,native_rmse=2*x)
              for s,i,x in [('a',1,1.),('a',2,3.),('b',1,8.)]])
        self.assertEqual(summarize(frame).iloc[0].mean_nrmse,5.)
        other = frame.copy()
        other['normalized_rmse'] += 1.
        result = contrast(frame,other,'toy')
        self.assertEqual(result['participants'],2)
        self.assertEqual((result['a_minus_b'],result['ci_low'],result['ci_high']),(-1.,-1.,-1.))
        self.assertEqual(result,contrast(frame,other,'toy'))


if __name__ == '__main__':
    unittest.main()
