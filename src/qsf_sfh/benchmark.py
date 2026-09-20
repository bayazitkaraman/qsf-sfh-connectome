"""Resumable consolidation with verified inputs and fixed analysis settings."""
from __future__ import annotations

import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

import numpy as np
import pandas as pd
from brain_quantum.analysis_suite import compute_walk_stack
from brain_quantum.dense_heat_capacity_control import dense_heat_kernels
from brain_quantum.methodology_controls import similarity_align
from brain_quantum.qrc_connectome import normalized_laplacian
from .core import FAMILIES, losses, reconstruct
from .data import check_release as check, load_design as check_previous, digest, dump, load_records, PROTOCOL, WORK
from .model import ENCODERS, POLICIES, fit, make_bundle, policy_choice, projection_diagnostics, robust_choice

OUT = WORK/'consolidation'
DATA = None


def qsf_stack(walk):
    return np.stack([v for values in zip(walk['q_real'], walk['q_imag'], walk['q_prob'])
                     for v in values]).astype(np.float32)


def prepare(cohort):
    check()
    folder = OUT/cohort/'data'
    if (folder/'COMPLETE.json').exists():
        verify(folder)
        return
    records = load_records(cohort)
    folder.mkdir(parents=True, exist_ok=False)
    n, nodes = len(records), len(records[0].weights)
    q = np.lib.format.open_memmap(folder/'qsf.npy', mode='w+', dtype='float32', shape=(n,18,nodes,nodes))
    h = np.lib.format.open_memmap(folder/'heat.npy', mode='w+', dtype='float32', shape=q.shape)
    for i, record in enumerate(records):
        q[i] = np.stack([v for values in zip(record.q_real, record.q_imag, record.q_prob) for v in values])
        h[i] = dense_heat_kernels(record, np.geomspace(.25,8,18).tolist())
    q.flush()
    h.flush()
    np.save(folder/'positions.npy', np.stack([r.connectome.positions for r in records]))
    np.save(folder/'extent.npy', np.array([r.extent for r in records]))
    np.save(folder/'weights.npy', np.stack([r.weights for r in records]))
    dump(folder/'subjects.json', [r.connectome.subject_id for r in records])
    dump(folder/'labels.json', records[0].connectome.node_names)
    previous = check_previous()
    for split in range(1,6):
        partitions = partition_indices(previous, cohort, split, [r.connectome.subject_id for r in records])
        fit_ids = partitions['fit']
        weights = np.mean(np.stack([records[i].weights for i in fit_ids]).astype(float), axis=0)
        walk = compute_walk_stack(normalized_laplacian(weights), [.25,.5,1,2,4,8])
        # The dense helper needs only the transformed weight matrix.
        from types import SimpleNamespace
        dense = dense_heat_kernels(SimpleNamespace(weights=weights), np.geomspace(.25,8,18).tolist())
        np.savez_compressed(folder/f'template_s{split}.npz', qsf=qsf_stack(walk), heat=dense,
                            weights=weights, fit_ids=np.array(fit_ids))
    finish(folder, dict(cohort=cohort, people=n, graph_nodes=nodes, input_graphs=len(records)))
    print(f'{cohort}: hash-verified features and fit-only templates prepared', flush=True)


def partition_indices(previous, cohort, split, subjects):
    result = previous['cohorts'][cohort]['partitions'][str(split)]
    assert len(set(subjects)) == len(subjects)
    assert sorted(sum([result[n] for n in ('fit','validation','test','calibration')], [])) == list(range(len(subjects)))
    return result


def finish(folder, extra):
    dump(folder/'COMPLETE.json', dict(**extra, protocol_sha256=digest(PROTOCOL),
         artifacts={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'COMPLETE.json'}))


def verify(folder):
    manifest = json.loads((folder/'COMPLETE.json').read_text())
    assert manifest['protocol_sha256'] == digest(PROTOCOL)
    for name, expected in manifest['artifacts'].items():
        assert digest(folder/name) == expected, str(folder/name)
    return manifest


def load_data(cohort):
    global DATA
    folder = OUT/cohort/'data'
    DATA = {name:np.load(folder/(name+'.npy'), mmap_mode='r')
            for name in ('qsf','heat','positions','extent','weights')}
    DATA['subjects'] = json.loads((folder/'subjects.json').read_text())
    DATA['templates'] = {s:dict(np.load(folder/f'template_s{s}.npz')) for s in range(1,6)}


def directory(spec):
    return OUT/spec['cohort']/'cells'/f"s{spec['split']}_m{spec['mask']}_k{spec['count']}_{spec['policy']}_{spec['encoder']}_{spec['graph']}_{spec['scaling']}"


def run_cell(spec):
    folder = directory(spec)
    if (folder/'COMPLETE.json').exists():
        manifest = verify(folder)
        assert manifest['spec'] == spec
        return
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    sources, targets = np.array(spec['sources']), np.array(spec['targets'])
    parts = spec['partitions']
    template = DATA['templates'][spec['split']] if spec['graph'] == 'template' else None
    if template is not None:
        assert list(template['fit_ids']) == parts['fit']
    inner = make_bundle(DATA, sources, targets, spec['encoder'], parts['fit'], template)
    final = make_bundle(DATA, sources, targets, spec['encoder'], parts['train'], template)
    grid, chosen, models, diagnostics = fit(inner, final, parts['fit'], parts['validation'],
                                           parts['train'], spec['count'], spec['scaling'])
    chosen = {f:dict(r, policy=spec['policy']) for f,r in chosen.items()}
    pd.DataFrame(grid).to_csv(folder/'validation_grid.csv', index=False)
    dump(folder/'selections.json', dict(families=chosen, robust=robust_choice(chosen), diagnostics=diagnostics))
    test = parts['test']
    xx = final.x[test]
    predictions, unprojected, rows, projection_rows = [], [], [], []
    model_arrays = {}
    for family in FAMILIES:
        row, model = chosen[family], models[family]
        raw = model.predict(xx.reshape(-1, xx.shape[-1])).reshape(len(test), len(targets), 6)
        prediction, clipped = reconstruct(raw, final, test, family, row['radius'])
        counterpart = 'ordinary' if family in ('ordinary','projection') else 'residual'
        before, _ = reconstruct(raw, final, test, counterpart, 0.)
        nr, nm = losses(prediction, final, test)
        before_nr, _ = losses(before, final, test)
        for local, i in enumerate(test):
            valid = final.finite[i]
            rows.append(dict(subject_id=DATA['subjects'][i], family=family,
                             normalized_rmse=float(nr[local]), normalized_mse=float(nm[local]),
                             unprojected_nrmse=float(before_nr[local]), target_count=int(valid.sum()),
                             clipped_targets=int((clipped[local]&valid).sum()),
                             center_fallback=int(final.fallback[i,0]), scale_fallback=int(final.fallback[i,1]),
                             response_fallback_targets=int(final.fallback[i,2]),
                             alpha=row['alpha'], radius=row['radius']))
        if family in ('projection','bounded_residual'):
            be, af, excess, inside = projection_diagnostics(before, prediction, final.y[test],
                  final.center[test], final.scale[test], row['radius'], final.finite[test], final.extent[test])
            for local, i in enumerate(test):
                valid = final.finite[i]
                for group, mask in (('all', valid), ('inside', valid&inside[local]), ('outside', valid&~inside[local])):
                    count = int(mask.sum())
                    projection_rows.append(dict(subject_id=DATA['subjects'][i], family=family, truth_group=group,
                        fallback=bool(final.fallback[i].any()), target_count=count,
                        clipped_targets=int((clipped[local]&mask).sum()),
                        beyond_distance_sum=float(excess[local,mask].sum()),
                        beyond_squared_sum=float(np.sum(excess[local,mask]**2)),
                        before_squared_sum=float(np.sum(be[local,mask]**2)),
                        after_squared_sum=float(np.sum(af[local,mask]**2)),
                        error_change_sum=float(np.sum(af[local,mask]-be[local,mask]))))
        predictions.append(prediction)
        unprojected.append(before)
        for key in ('mean','scale','transformed_mean','y_mean','coef'):
            model_arrays[family+'_'+key] = getattr(model,key)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv', index=False)
    pd.DataFrame(projection_rows).to_csv(folder/'projection.csv', index=False)
    np.savez_compressed(folder/'models.npz', **model_arrays)
    np.savez_compressed(folder/'predictions.npz', prediction=np.stack(predictions), unprojected=np.stack(unprojected),
                        truth=final.y[test], finite=final.finite[test], extent=final.extent[test],
                        center=final.center[test], scale=final.scale[test], fallback=final.fallback[test],
                        targets=targets, sources=sources,
                        subject_ids=np.array([DATA['subjects'][i] for i in test]))
    if spec['encoder'] == 'qsf' and spec['graph'] == 'individual' and spec['scaling'] == 'tanh3':
        atlas_predictions(folder, sources, targets, parts)
    finish(folder, dict(spec=spec, seconds=time.monotonic()-start))


def atlas_predictions(folder, sources, targets, parts):
    pos = DATA['positions'][parts['train']]
    finite = np.isfinite(pos).all(axis=2)
    sums = np.where(finite[:,:,None],pos,0.).sum(axis=0)
    counts = finite.sum(axis=0)[:,None]
    atlas = np.divide(sums, counts, out=np.full_like(sums,np.nan), where=counts>0)
    test = parts['test']
    truth = DATA['positions'][test][:, targets]
    valid = np.isfinite(truth).all(axis=2)
    predictions = [np.broadcast_to(atlas[targets], truth.shape).copy(), []]
    for i in test:
        source_only = np.full_like(atlas, np.nan)
        source_only[sources] = DATA['positions'][i,sources]
        predictions[1].append(similarity_align(atlas, source_only, sources)[targets])
    predictions[1] = np.stack(predictions[1])
    rows = []
    for method, pred in zip(('atlas_mean','atlas_aligned'),predictions):
        if not np.isfinite(pred[valid]).all():
            raise ValueError('Atlas unavailable for an evaluated target; do not silently drop it.')
        nm = np.where(valid,np.sum((pred-truth)**2,axis=2),0.).sum(axis=1)/valid.sum(axis=1)/DATA['extent'][test]**2
        rows.extend(dict(subject_id=DATA['subjects'][i],family=method,normalized_rmse=float(np.sqrt(nm[l])),
                         normalized_mse=float(nm[l]),target_count=int(valid[l].sum())) for l,i in enumerate(test))
    pd.DataFrame(rows).to_csv(folder/'atlas_metrics.csv', index=False)
    np.savez_compressed(folder/'atlas_predictions.npz', prediction=np.stack(predictions), truth=truth,
                        finite=valid, extent=DATA['extent'][test])


def specs(cohort, phase):
    previous = check_previous()
    subjects = json.loads((OUT/cohort/'data/subjects.json').read_text())
    result = []
    for split in range(1,6):
        partitions = partition_indices(previous, cohort, split, subjects)
        for mask, targets in enumerate(previous['target_masks'][cohort],1):
            rankings = previous['cohorts'][cohort]['rankings'][f's{split}_m{mask}']
            for count in (8,16,32):
                for encoder in ENCODERS:
                    policies = list(POLICIES)+['spatial'] if phase == 'main' else ['random']
                    base = dict(cohort=cohort,split=split,mask=mask,count=count,encoder=encoder,
                                graph='individual',scaling='tanh3',targets=targets,partitions=partitions)
                    if phase == 'template':
                        choices = []
                        for policy in POLICIES:
                            s = dict(base,policy=policy,sources=rankings[policy][:count])
                            choices.append(json.loads((directory(s)/'selections.json').read_text())['robust'])
                        policies = sorted(set(['random',policy_choice(choices)['policy']]))
                        base['graph'] = 'template'
                    if phase == 'sensitivity':
                        base['scaling'] = 'group_floor'
                    for policy in policies:
                        source = rankings[policy][:count]
                        assert not set(source)&set(targets) and len(set(source)) == count
                        result.append(dict(base,policy=policy,sources=source))
    return result


def run(cohort, phase, workers):
    frozen = check()
    verify(OUT/cohort/'data')
    jobs = specs(cohort,phase)
    manifest = OUT/cohort/(phase+'_PLAN.json')
    if manifest.exists():
        assert json.loads(manifest.read_text())['jobs'] == jobs
    else:
        dump(manifest,dict(jobs=jobs,protocol_sha256=frozen['protocol_sha256']))
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=load_data,initargs=(cohort,)) as pool:
        pending = [pool.submit(run_cell,job) for job in jobs]
        for number,future in enumerate(as_completed(pending),1):
            future.result()
            if number%30 == 0 or number == len(jobs):
                print(f'{cohort} {phase}: {number}/{len(jobs)}, elapsed {time.monotonic()-start:.0f}s',flush=True)
    check()
    dump(OUT/cohort/(phase+'_COMPLETE.json'),dict(jobs=len(jobs),seconds=time.monotonic()-start,
         protocol_sha256=frozen['protocol_sha256'],cells={str(directory(j).relative_to(OUT)):digest(directory(j)/'COMPLETE.json') for j in jobs}))

