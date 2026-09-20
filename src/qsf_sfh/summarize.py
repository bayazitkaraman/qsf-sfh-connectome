"""Reconstruct every consolidation endpoint and report prespecified contrasts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from .benchmark import (FAMILIES, ENCODERS, POLICIES, OUT, check, directory,
                        dump, verify)
from .model import choose_families, policy_choice, projection_diagnostics, robust_choice
import numpy as np
import pandas as pd

KEY = ['split_id','mask_id','subject_id']


def summary(frame):
    participant = frame.groupby('subject_id')[['normalized_rmse','normalized_mse']].mean()
    return dict(equal_participant_nrmse=float(participant.normalized_rmse.mean()),
                equal_participant_nmse=float(participant.normalized_mse.mean()),
                row_mean_nrmse=float(frame.normalized_rmse.mean()),
                p95=float(frame.normalized_rmse.quantile(.95)),
                p99=float(frame.normalized_rmse.quantile(.99)),maximum=float(frame.normalized_rmse.max()),
                rows_above_one=int((frame.normalized_rmse>1).sum()),
                participants=len(participant),records=len(frame))


def audit_cell(folder):
    manifest = verify(folder)
    spec = manifest['spec']
    choices = json.loads((folder/'selections.json').read_text())
    replay = choose_families(pd.read_csv(folder/'validation_grid.csv').to_dict('records'))
    for family in FAMILIES:
        for key in ('alpha','radius'):
            assert replay[family][key] == choices['families'][family][key]
    assert robust_choice(choices['families']) == choices['robust']
    metrics = pd.read_csv(folder/'metrics.csv',dtype={'subject_id':str})
    with np.load(folder/'predictions.npz') as p:
        assert list(p['sources']) == spec['sources']
        assert list(p['targets']) == spec['targets']
        assert not set(p['sources'])&set(p['targets'])
        truth, valid, extent = p['truth'],p['finite'],p['extent']
        assert np.array_equal(valid,np.isfinite(truth).all(axis=2))
        for index,family in enumerate(FAMILIES):
            pred = p['prediction'][index]
            assert np.isfinite(pred).all()
            nm = np.where(valid,np.sum((pred-truth)**2,axis=2),0.).sum(axis=1)/valid.sum(axis=1)/extent**2
            rows = metrics[metrics.family==family].set_index('subject_id').loc[p['subject_ids']]
            np.testing.assert_allclose(np.sqrt(nm),rows.normalized_rmse,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(nm,rows.normalized_mse,rtol=1e-12,atol=1e-12)
            if family in ('projection','bounded_residual'):
                radius = choices['families'][family]['radius']
                bound = radius*p['scale'][:,None]
                assert np.all(np.linalg.norm(pred-p['center'],axis=2) <= bound+1e-8)
                projection_diagnostics(p['unprojected'][index],pred,truth,p['center'],p['scale'],radius,valid,extent)
    if (folder/'atlas_predictions.npz').exists():
        atlas_rows = pd.read_csv(folder/'atlas_metrics.csv')
        with np.load(folder/'atlas_predictions.npz') as a:
            for i,family in enumerate(('atlas_mean','atlas_aligned')):
                nm = np.where(a['finite'],np.sum((a['prediction'][i]-a['truth'])**2,axis=2),0.).sum(axis=1)/a['finite'].sum(axis=1)/a['extent']**2
                np.testing.assert_allclose(nm,atlas_rows[atlas_rows.family==family].normalized_mse,rtol=1e-12,atol=1e-12)
    return spec,choices,metrics


def paired(frame,a,b,cohort,count,encoder):
    subset = frame[(frame.cohort==cohort)&(frame.source_count==count)&(frame.encoder==encoder)]
    left = subset[subset.track==a].set_index(KEY)
    right = subset[subset.track==b].set_index(KEY)
    assert left.index.is_unique and right.index.is_unique
    assert set(left.index) == set(right.index),(a,b,cohort,count,encoder)
    difference = (left[['normalized_rmse','normalized_mse']]-right[['normalized_rmse','normalized_mse']]).groupby('subject_id').mean()
    seed = int(hashlib.sha256(f'{a}|{b}|{cohort}|{count}|{encoder}'.encode()).hexdigest()[:8],16)
    rng = np.random.default_rng(seed)
    results = []
    for metric in ('normalized_rmse','normalized_mse'):
        values = difference[metric].to_numpy()
        means = []
        for _ in range(50):
            means.extend(values[rng.integers(0,len(values),(100,len(values)))].mean(axis=1))
        results.append(dict(cohort=cohort,source_count=count,encoder=encoder,contrast=a+' minus '+b,
                            metric=metric,paired_difference=float(values.mean()),
                            ci95_low=float(np.quantile(means,.025)),ci95_high=float(np.quantile(means,.975)),
                            participant_win_fraction=float((values<0).mean()),participants=len(values)))
    return results


def main():
    frozen = check()
    track_frames,all_choices, projection_parts,candidate_groups = [],[],[],{}
    verified_cells,verified_rows = 0,0
    for cohort in ('hcp83','oasis'):
        registry = {}
        for phase in ('main','template','sensitivity'):
            completion = json.loads((OUT/cohort/(phase+'_COMPLETE.json')).read_text())
            assert completion['protocol_sha256'] == frozen['protocol_sha256']
            for relative, expected in completion['cells'].items():
                folder = OUT/relative
                assert hashlib.sha256((folder/'COMPLETE.json').read_bytes()).hexdigest() == expected
                spec,choices,metrics = audit_cell(folder)
                verified_cells += 1
                verified_rows += len(metrics)
                key = tuple(spec[k] for k in ('split','mask','count','encoder','policy','graph','scaling'))
                registry[key] = (folder,spec,choices)
                for family,part in metrics.groupby('family'):
                    candidate_key = (cohort,spec['count'],spec['encoder'],spec['policy'],spec['graph'],spec['scaling'],family)
                    candidate_groups.setdefault(candidate_key,[]).append(part[['subject_id','normalized_rmse','normalized_mse']])
                projection = pd.read_csv(folder/'projection.csv',dtype={'subject_id':str})
                for name,value in [('cohort',cohort),('source_count',spec['count']),('encoder',spec['encoder']),
                                   ('policy',spec['policy']),('graph',spec['graph']),('scaling',spec['scaling'])]:
                    projection[name] = value
                projection_parts.append(projection)
                if verified_cells%300 == 0:
                    print(f'Audited {verified_cells} cells / {verified_rows} participant-family records',flush=True)

        def cell(split,mask,count,encoder,policy,graph='individual',scaling='tanh3'):
            return registry[(split,mask,count,encoder,policy,graph,scaling)]

        def add(entry,track,encoder,family=None,atlas=False):
            folder,spec,choices = entry
            if family is None:
                family = choices['robust']['family']
            frame = pd.read_csv(folder/('atlas_metrics.csv' if atlas else 'metrics.csv'),dtype={'subject_id':str})
            frame = frame[frame.family==family].copy()
            assert len(frame)
            frame['cohort'],frame['encoder'],frame['track'] = cohort,encoder,track
            frame['split_id'],frame['mask_id'],frame['source_count'] = spec['split'],spec['mask'],spec['count']
            frame['policy'],frame['graph'],frame['scaling'] = spec['policy'],spec['graph'],spec['scaling']
            track_frames.append(frame)

        for split in range(1,6):
            for mask in range(1,4):
                for count in (8,16,32):
                    for encoder in ENCODERS:
                        entries = [cell(split,mask,count,encoder,p) for p in POLICIES]
                        selected = policy_choice([e[2]['robust'] for e in entries])
                        ordinary = policy_choice([e[2]['families']['ordinary'] for e in entries])
                        base = cell(split,mask,count,encoder,'random')
                        chosen = cell(split,mask,count,encoder,selected['policy'])
                        add(base,'random_robust',encoder)
                        add(chosen,'selected_robust',encoder)
                        add(cell(split,mask,count,encoder,'spatial'),'spatial_robust',encoder)
                        add(base,'random_ordinary',encoder,'ordinary')
                        add(chosen,'selected_sources_ordinary',encoder,'ordinary')
                        add(cell(split,mask,count,encoder,ordinary['policy']),'ordinary_selected',encoder,'ordinary')
                        add(cell(split,mask,count,encoder,'random',scaling='group_floor'),'floor_random_robust',encoder)
                        for policy,label in [('random','random'),(selected['policy'],'selected')]:
                            template = cell(split,mask,count,encoder,policy,'template')
                            add(template,'template_'+label+'_robust',encoder)
                            for family in ('projection','bounded_residual'):
                                add(template,'template_'+label+'_'+family,encoder,family)
                                add(cell(split,mask,count,encoder,policy),label+'_'+family,encoder,family)
                            atlas = cell(split,mask,count,'qsf',policy)
                            add(atlas,'atlas_'+label+'_aligned',encoder,'atlas_aligned',True)
                            add(atlas,'atlas_'+label+'_mean',encoder,'atlas_mean',True)
                        all_choices.append(dict(cohort=cohort,split_id=split,mask_id=mask,source_count=count,
                                                encoder=encoder,**selected))
        if cohort == 'hcp83':
            print('HCP audit and selection reconstruction complete',flush=True)
    tracks = pd.concat(track_frames,ignore_index=True)
    tracks.to_csv(OUT/'track_metrics.csv',index=False)
    pd.DataFrame(all_choices).to_csv(OUT/'selected_choices.csv',index=False)
    group_names = ['cohort','source_count','encoder','track']
    summaries = [dict(zip(group_names,key),**summary(group)) for key,group in tracks.groupby(group_names)]
    pd.DataFrame(summaries).to_csv(OUT/'track_summary.csv',index=False)
    split_names = group_names+['split_id','mask_id']
    pd.DataFrame([dict(zip(split_names,key),**summary(group)) for key,group in tracks.groupby(split_names)]).to_csv(OUT/'split_mask_summary.csv',index=False)
    candidate_names = ['cohort','source_count','encoder','policy','graph','scaling','family']
    candidates = [dict(zip(candidate_names,key),**summary(pd.concat(parts,ignore_index=True))) for key,parts in candidate_groups.items()]
    pd.DataFrame(candidates).to_csv(OUT/'candidate_summary.csv',index=False)
    projection = pd.concat(projection_parts,ignore_index=True)
    projection_keys = candidate_names+['truth_group','fallback']
    sums = ['target_count','clipped_targets','beyond_distance_sum','beyond_squared_sum','before_squared_sum','after_squared_sum','error_change_sum']
    projection.groupby(projection_keys,dropna=False)[sums].sum().reset_index().to_csv(OUT/'projection_summary.csv',index=False)

    comparisons = []
    pairs = [('selected_robust','random_robust'),('random_robust','random_ordinary'),
             ('ordinary_selected','random_ordinary'),('selected_sources_ordinary','random_ordinary'),
             ('selected_robust','selected_sources_ordinary'),('selected_robust','random_ordinary'),
             ('spatial_robust','selected_robust'),('floor_random_robust','random_robust'),
             ('random_robust','template_random_robust'),('selected_robust','template_selected_robust'),
             ('random_projection','random_bounded_residual'),
             ('random_robust','atlas_random_aligned'),('selected_robust','atlas_selected_aligned')]
    for label in ('random','selected'):
        for family in ('projection','bounded_residual'):
            pairs.append((label+'_'+family,'template_'+label+'_'+family))
    for cohort in ('hcp83','oasis'):
        for count in (8,16,32):
            for encoder in ENCODERS:
                for a,b in pairs:
                    comparisons.extend(paired(tracks,a,b,cohort,count,encoder))
            for track in ('random_robust','selected_robust'):
                for heat in ('heat_raw18','heat_norm18'):
                    subset = tracks[(tracks.track==track)&tracks.encoder.isin(['qsf',heat])].copy()
                    subset['track'] = subset.encoder
                    subset['encoder'] = 'matched_representation'
                    result = paired(subset,'qsf',heat,cohort,count,'matched_representation')
                    for row in result:
                        row['contrast'] = track+': '+row['contrast']
                    comparisons.extend(result)
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)
    dump(OUT/'VERIFICATION.json',dict(status='PASS',cells=verified_cells,
         participant_family_records=verified_rows,protocol_sha256=frozen['protocol_sha256'],
         code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         checks=['archive hashes','selections replay','all metric reconstructions','finite protected targets',
                 'source-target separation','mandatory bounds','identical-model projection inequalities',
                 'atlas metric reconstruction','paired participant alignment'],
         scope='Exploratory; reused participants; bootstrap conditional on fitted splits.'))
    print('All consolidation audits and paired summaries complete',flush=True)


if __name__ == '__main__':
    main()
