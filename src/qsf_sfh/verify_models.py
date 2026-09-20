"""Independent real-data solver and template checks, without refitting outcomes."""
import sys
from pathlib import Path

from . import benchmark as ex
import argparse
import json
import numpy as np
from brain_quantum.decoder_robustness import transform
from .model import make_bundle


def main(cohorts=('hcp83','oasis')):
    ex.check()
    reports = []
    for cohort in cohorts:
        ex.load_data(cohort)
        for split,template in ex.DATA['templates'].items():
            expected = np.asarray(ex.DATA['weights'][template['fit_ids']],float).mean(axis=0)
            np.testing.assert_array_equal(expected,template['weights'])
        for encoder in ex.ENCODERS:
            for scaling in ('tanh3','group_floor'):
                folder = ex.OUT/cohort/'cells'/f's2_m3_k8_random_{encoder}_individual_{scaling}'
                spec = json.loads((folder/'COMPLETE.json').read_text())['spec']
                parts = spec['partitions']
                bundle = make_bundle(ex.DATA,np.array(spec['sources']),np.array(spec['targets']),encoder,parts['train'])
                x,y = bundle.fit_arrays(parts['train'])
                selection = json.loads((folder/'selections.json').read_text())['families']['bounded_residual']
                with np.load(folder/'models.npz') as m:
                    prefix = 'bounded_residual_'
                    z = (x-m[prefix+'mean'])/m[prefix+'scale']
                    if scaling == 'tanh3':
                        z = transform(z,'tanh3')
                    z -= m[prefix+'transformed_mean']
                    alpha = selection['alpha']
                    a = np.vstack([z,np.sqrt(alpha)*np.eye(z.shape[1])])
                    b = np.vstack([y-m[prefix+'y_mean'],np.zeros((z.shape[1],6))])
                    coef = np.linalg.lstsq(a,b,rcond=None)[0]
                    relative = float(np.linalg.norm(coef-m[prefix+'coef'])/max(np.linalg.norm(coef),1e-30))
                    np.testing.assert_allclose(coef,m[prefix+'coef'],rtol=1e-7,atol=1e-7)
                    xt = bundle.x[parts['test']]
                    zt = (xt-m[prefix+'mean'])/m[prefix+'scale']
                    if scaling == 'tanh3':
                        zt = transform(zt,'tanh3')
                    raw = (zt-m[prefix+'transformed_mean'])@coef+m[prefix+'y_mean']
                    prediction = bundle.center[parts['test']]+bundle.scale[parts['test'],None,None]*raw[...,3:]
                    with np.load(folder/'predictions.npz') as p:
                        expected_prediction = p['unprojected'][3]
                        pred_relative = float(np.linalg.norm(prediction-expected_prediction)/max(np.linalg.norm(prediction),1e-30))
                        np.testing.assert_allclose(prediction,expected_prediction,rtol=1e-7,atol=1e-7)
                reports.append(dict(cohort=cohort,encoder=encoder,scaling=scaling,
                                    coefficient_relative_error=relative,prediction_relative_error=pred_relative))
    suffix = '' if len(cohorts)==2 else '_'+cohorts[0]
    ex.dump(ex.OUT/('INDEPENDENT_SOLVER_CHECK'+suffix+'.json'),
            dict(status='PASS',fits=reports,templates_checked=5*len(cohorts)))
    print(json.dumps(reports,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=('hcp83','oasis'))
    args = parser.parse_args()
    main((args.cohort,) if args.cohort else ('hcp83','oasis'))
