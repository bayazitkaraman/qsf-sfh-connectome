"""Describe clipping by scored-target fallback, not by an entire mask's flag."""
import sys
from pathlib import Path

from . import benchmark as ex
import json
import numpy as np
import pandas as pd
from .model import projection_diagnostics


def main():
    ex.check()
    rows = []
    for cohort in ('hcp83','oasis'):
        ex.load_data(cohort)
        folders = sorted((ex.OUT/cohort/'cells').glob('s*_m*_k*_qsf_individual_tanh3'))
        for folder in folders:
            manifest = ex.verify(folder)
            spec = manifest['spec']
            choices = json.loads((folder/'selections.json').read_text())['families']
            with np.load(folder/'predictions.npz') as p:
                no_response = []
                for local,index in enumerate(spec['partitions']['test']):
                    source_valid = np.isfinite(ex.DATA['positions'][index,p['sources']]).all(axis=1)
                    q = ex.DATA['qsf'][index][2::3][:,p['targets']][:,:,p['sources']]
                    response = np.asarray(q.mean(axis=0),float)
                    missing = np.maximum(response[:,source_valid],0.).sum(axis=1) <= 0
                    assert int(missing.sum()) == p['fallback'][local,2]
                    no_response.append(missing)
                no_response = np.stack(no_response)
                fallback = no_response | (p['fallback'][:,:2].any(axis=1))[:,None]
                for index,family in ((1,'projection'),(3,'bounded_residual')):
                    radius = choices[family]['radius']
                    before,after,excess,inside = projection_diagnostics(
                        p['unprojected'][index],p['prediction'][index],p['truth'],p['center'],p['scale'],
                        radius,p['finite'],p['extent'])
                    clipped = np.linalg.norm(p['unprojected'][index]-p['center'],axis=2) > radius*p['scale'][:,None]
                    for name,truth_mask in (('all',p['finite']),('inside',p['finite']&inside),('outside',p['finite']&~inside)):
                        for flag in (False,True):
                            valid = truth_mask & (fallback==flag)
                            rows.append(dict(cohort=cohort,split_id=spec['split'],mask_id=spec['mask'],
                                source_count=spec['count'],policy=spec['policy'],family=family,
                                truth_group=name,scored_target_fallback=flag,target_count=int(valid.sum()),
                                clipped_targets=int((clipped&valid).sum()),
                                beyond_distance_sum=float(excess[valid].sum()),
                                before_squared_sum=float(np.sum(before[valid]**2)),
                                after_squared_sum=float(np.sum(after[valid]**2)),
                                error_change_sum=float((after-before)[valid].sum())))
        print(cohort,'scored-target fallback audit:',len(folders),'QSF cells',flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ex.OUT/'projection_target_cells.csv',index=False)
    keys = ['cohort','source_count','family','truth_group','scored_target_fallback']
    values = ['target_count','clipped_targets','beyond_distance_sum','before_squared_sum','after_squared_sum','error_change_sum']
    result = frame.groupby(keys)[values].sum().reset_index()
    result.to_csv(ex.OUT/'projection_target_summary.csv',index=False)
    old = pd.read_csv(ex.OUT/'projection_summary.csv')
    old = old[(old.graph=='individual')&(old.scaling=='tanh3')&(old.encoder=='qsf')]
    common = ['cohort','source_count','family','truth_group']
    for field in values:
        np.testing.assert_allclose(result.groupby(common)[field].sum(),old.groupby(common)[field].sum(),
                                   rtol=1e-10,atol=1e-8)
    ex.dump(ex.OUT/'PROJECTION_TARGET_AUDIT.json',dict(status='PASS',qsf_cells=630,
        checks=['no-response flags replayed','same predictions','same total errors and clipping',
                'fallback strata restricted to scored targets'],
        description='Postprocessing refinement only; whole-mask flags remain in original archives.'))


if __name__ == '__main__':
    main()
