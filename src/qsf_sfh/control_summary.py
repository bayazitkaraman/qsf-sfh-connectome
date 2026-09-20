"""Audit all additional controls and summarize paired participant errors."""
import hashlib
import json

import numpy as np
import pandas as pd
from . import controls as c
from .core import Bundle, reconstruct
from .data import check_release, load_design
from .model import choose_families, robust_choice

KEY = ['split', 'mask', 'subject_id']


def audit(spec):
    folder = c.directory(spec)
    manifest = c.verify(folder)
    assert manifest['spec'] == spec
    selected = json.loads((folder/'selections.json').read_text())
    grid = pd.read_csv(folder/'validation_grid.csv',float_precision='round_trip').to_dict('records')
    replay = robust_choice(choose_families(grid))
    for key in ('family','alpha','radius'):
        assert replay[key] == selected['robust'][key]
    p = dict(np.load(folder/'predictions.npz'))
    n = len(p['prediction'])
    b = Bundle(np.zeros((n,1,1)),p['truth'],p['finite'],p['extent'],p['center'],p['scale'],np.zeros(n),np.zeros((n,3)))
    prediction, clipped = reconstruct(p['raw'],b,list(range(n)),replay['family'],replay['radius'])
    np.testing.assert_allclose(prediction,p['prediction'],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(clipped,p['clipped'])
    nr,nm = c.losses(prediction,b,list(range(n)))
    metrics = pd.read_csv(folder/'metrics.csv',dtype={'subject_id':str},float_precision='round_trip')
    np.testing.assert_array_equal(metrics.subject_id,p['subject_ids'])
    np.testing.assert_allclose(metrics.normalized_rmse,nr,atol=1e-12,rtol=0)
    np.testing.assert_allclose(metrics.normalized_mse,nm,atol=1e-12,rtol=0)
    np.testing.assert_allclose(metrics.native_rmse,nr*p['extent'],atol=1e-10,rtol=0)
    np.testing.assert_array_equal(p['targets'],spec['targets'])
    np.testing.assert_array_equal(p['sources'],spec['sources'])
    assert not set(spec['sources'])&set(spec['targets'])
    if spec['task'] == 'unseen':
        assert not set(selected['diagnostics']['training_labels'])&set(spec['targets']+spec['sources'])
    for key in ('cohort','split','mask','count','task','encoder'):
        metrics[key] = spec[key]
    return metrics, manifest


def contrast(a,b,name):
    assert not a.duplicated(KEY).any() and not b.duplicated(KEY).any()
    aa = a.set_index(KEY).normalized_rmse.sort_index()
    bb = b.set_index(KEY).normalized_rmse.sort_index()
    assert aa.index.equals(bb.index)
    diff = (aa-bb).groupby('subject_id').mean().to_numpy()
    am = aa.groupby('subject_id').mean().mean()
    bm = bb.groupby('subject_id').mean().mean()
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
    rng = np.random.default_rng(seed)
    boots = np.concatenate([rng.choice(diff,size=(100,len(diff)),replace=True).mean(axis=1) for _ in range(50)])
    low,high = np.quantile(boots,[.025,.975])
    return dict(comparison=name,a_mean=float(am),b_mean=float(bm),a_minus_b=float(diff.mean()),
                ci_low=float(low),ci_high=float(high),a_reduction_percent=float(100*(bm-am)/bm),
                a_win_fraction=float(np.mean(diff<0)),participants=len(diff),records=len(aa))


def summarize(frame):
    keys = ['cohort','task','count','encoder']
    person = frame.groupby(keys+['subject_id'])[['normalized_rmse','normalized_mse','native_rmse']].mean().reset_index()
    rows = []
    for key, group in person.groupby(keys,sort=True):
        rows.append(dict(zip(keys,key),mean_nrmse=float(group.normalized_rmse.mean()),
                    mean_nmse=float(group.normalized_mse.mean()),native_rmse=float(group.native_rmse.mean()),
                    median_person_nrmse=float(group.normalized_rmse.median()),
                    p95_person_nrmse=float(group.normalized_rmse.quantile(.95)),participants=len(group)))
    return pd.DataFrame(rows)


def comparisons(frame):
    pairs = []
    for cohort in ('hcp83','oasis'):
        for count in (8,16,32):
            base = frame[(frame.cohort==cohort)&(frame['count']==count)]
            for task,encoders in (('known',c.KNOWN),('unseen',c.UNSEEN)):
                data = base[base.task==task]
                for encoder in encoders:
                    if encoder == 'qsf':
                        continue
                    row = contrast(data[data.encoder=='qsf'],data[data.encoder==encoder],f'{cohort}_{count}_{task}_qsf_{encoder}')
                    pairs.append(dict(cohort=cohort,count=count,task=task,a='qsf',b=encoder,**row))
            for encoder in ('qsf','heat_raw18','heat_norm18','shortest_path'):
                data = base[base.encoder==encoder]
                row = contrast(data[data.task=='unseen'],data[data.task=='known'],f'{cohort}_{count}_unseen_vs_known_{encoder}')
                pairs.append(dict(cohort=cohort,count=count,task='unseen_vs_known',a=encoder+'_unseen',b=encoder+'_known',**row))
    return pd.DataFrame(pairs)


def main():
    check_release()
    plans = c.plans(load_design())
    frames, manifests = [], []
    for i, spec in enumerate(plans,1):
        frame, manifest = audit(spec)
        frames.append(frame)
        manifests.append(manifest)
        if i%100 == 0:
            print(f'Audited {i}/{len(plans)} controls',flush=True)
    frame = pd.concat(frames,ignore_index=True)
    frame.to_csv(c.OUT/'participant_metrics.csv',index=False)
    summary = summarize(frame)
    summary.to_csv(c.OUT/'controls_summary.csv',index=False)
    paired = comparisons(frame)
    paired.to_csv(c.OUT/'controls_comparisons.csv',index=False)
    independent = [m['independent_solver_max_abs_error'] for m in manifests if m['independent_solver_max_abs_error'] is not None]
    poison = [m['hidden_coordinate_poison_passed'] for m in manifests if m['hidden_coordinate_poison_passed'] is not None]
    assert len(independent)>=28 and len(poison)>=10 and all(poison)
    verification = dict(completed_cells=len(manifests),participant_metric_rows=len(frame),paired_comparisons=len(paired),
                        independent_solver_checks=len(independent),max_solver_difference=max(independent),
                        hidden_coordinate_checks=len(poison),all_hidden_coordinate_checks_passed=all(poison))
    c.dump(c.OUT/'controls_verification.json',verification)
    print(summary[summary['count']==32].to_string(index=False))
    print(json.dumps(verification,indent=2))
