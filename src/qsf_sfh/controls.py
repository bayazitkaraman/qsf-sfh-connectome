"""Channel, shortest-path, and unseen-region controls on the fixed cohort design."""
from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from brain_quantum.analysis_suite import (all_pairs_shortest_path, compute_walk_stack,
                                        position_extent, response_coordinate_moments)
from brain_quantum.decoder_robustness import transform
from brain_quantum.qrc_connectome import normalized_laplacian
from brain_quantum.source_coordinate_v2 import response_scale
from .core import Bundle, FAMILIES, losses, reconstruct, source_frame
from . import benchmark as original
from .model import features as original_features, geometry_prior, robust_choice
from .control_solver import fit
from .data import ROOT, WORK, check_release, load_design

PROTOCOL = ROOT/'configs/controls_protocol.txt'
OUT = WORK/'controls'
KNOWN = ('qsf', 'heat_raw18', 'heat_norm18', 'prob6', 'amp12', 'prob18', 'amp18', 'shortest_path', 'moments')
UNSEEN = ('qsf', 'heat_raw18', 'heat_norm18', 'shortest_path', 'template_qsf')
DATA = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def plans(previous):
    result = []
    for cohort in ('hcp83', 'oasis'):
        for split in range(1, 6):
            for mask, targets in enumerate(previous['target_masks'][cohort], 1):
                ranking = previous['cohorts'][cohort]['rankings'][f's{split}_m{mask}']['random']
                for count in (8, 16, 32):
                    sources = ranking[:count]
                    assert len(sources) == count and not set(sources) & set(targets)
                    for task, encoders in (('known', KNOWN), ('unseen', UNSEEN)):
                        for encoder in encoders:
                            result.append(dict(cohort=cohort, split=split, mask=mask, count=count,
                                               task=task, encoder=encoder, sources=sources, targets=targets))
    assert len(result) == 1260
    return result


def finish(folder, **extra):
    dump(folder/'COMPLETE.json', dict(**extra, protocol_sha256=digest(PROTOCOL),
         artifacts={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'COMPLETE.json'}))


def verify(folder):
    manifest = json.loads((folder/'COMPLETE.json').read_text())
    assert manifest['protocol_sha256'] == digest(PROTOCOL)
    for name, expected in manifest['artifacts'].items():
        assert digest(folder/name) == expected, str(folder/name)
    return manifest


def prepare(cohort):
    check_release()
    original.verify(original.OUT/cohort/'data')
    folder = OUT/cohort/'data'
    if (folder/'COMPLETE.json').exists():
        verify(folder)
        return
    folder.mkdir(parents=True, exist_ok=False)
    w = np.load(original.OUT/cohort/'data/weights.npy', mmap_mode='r')
    n, nodes, _ = w.shape
    p = np.lib.format.open_memmap(folder/'prob18.npy', mode='w+', dtype='float32', shape=(n,18,nodes,nodes))
    a = np.lib.format.open_memmap(folder/'amp18.npy', mode='w+', dtype='float32', shape=p.shape)
    d = np.lib.format.open_memmap(folder/'shortest_path.npy', mode='w+', dtype='float32', shape=w.shape)
    for i in range(n):
        lap = normalized_laplacian(w[i])
        p[i] = np.stack(compute_walk_stack(lap, np.geomspace(.25,8,18).tolist())['q_prob'])
        walk = compute_walk_stack(lap, np.geomspace(.25,8,9).tolist())
        a[i] = np.stack([v for values in zip(walk['q_real'],walk['q_imag']) for v in values])
        d[i] = all_pairs_shortest_path(w[i])
        if (i+1) % 100 == 0:
            print(f'{cohort}: prepared {i+1}/{n}', flush=True)
    p.flush()
    a.flush()
    d.flush()
    finish(folder, cohort=cohort, people=n, fully_edgeless=int(np.sum(~np.any(w>0,axis=(1,2)))))


def load_data(cohort):
    global DATA
    original.load_data(cohort)
    DATA = dict(original.DATA)
    for name in ('prob18', 'amp18', 'shortest_path'):
        DATA[name] = np.load(OUT/cohort/'data'/(name+'.npy'), mmap_mode='r')


def features(data, i, sources, targets, encoder, source_positions, split):
    if encoder in ('qsf', 'heat_raw18', 'heat_norm18', 'template_qsf'):
        base_encoder = 'qsf' if encoder == 'template_qsf' else encoder
        stack_name = 'qsf' if base_encoder == 'qsf' else 'heat'
        stack = data['templates'][split][stack_name] if encoder == 'template_qsf' else data[stack_name][i]
        return original_features(stack, source_positions, sources, targets, base_encoder)
    if encoder == 'shortest_path':
        distances = data['shortest_path'][i][:, sources]
        scaled = distances/response_scale(distances)
        response = np.exp(-scaled)[targets]
        base = np.concatenate([scaled[targets], response], axis=1)
    else:
        selected = data['qsf'][i][:, targets][:, :, sources]
        response = selected[2::3].mean(axis=0)
        if encoder == 'prob6':
            dynamic = selected[2::3]
        elif encoder == 'amp12':
            dynamic = selected[np.arange(18) % 3 != 2]
        elif encoder in ('prob18', 'amp18'):
            dynamic = data[encoder][i][:, targets][:, :, sources]
        elif encoder == 'moments':
            dynamic = selected[:0]
        else:
            raise ValueError(encoder)
        base = dynamic.transpose(1,0,2).reshape(len(targets), dynamic.shape[0]*len(sources))
    mean, spread = response_coordinate_moments(response, source_positions)
    return np.concatenate([base, mean, spread],axis=1).astype(np.float32), response


def learning_view(data, spec):
    positions = np.array(data['positions'], copy=True)
    hidden = np.array(spec['targets'])
    if spec['task'] == 'unseen':
        positions[:, hidden] = np.nan
        extent = np.array([position_extent(p) for p in positions])
        training_targets = np.array(sorted(set(range(positions.shape[1]))-set(hidden)-set(spec['sources'])))
    else:
        extent = data['extent']
        training_targets = hidden
    return positions, extent, training_targets


def bundle(data, spec, targets, training, positions, extent):
    sources = np.array(spec['sources'])
    prior, radius = geometry_prior(positions, sources, training)
    xx, bb, ss, ff = [], [], [], []
    for i in range(len(positions)):
        x, response = features(data, i, sources, targets, spec['encoder'], positions[i,sources], spec['split'])
        center, scale, no_sources, small_radius, no_response = source_frame(response, positions[i,sources], prior, radius)
        xx.append(x)
        bb.append(center)
        ss.append(scale)
        ff.append([no_sources, small_radius, no_response])
    y = positions[:, targets]
    return Bundle(np.stack(xx), y, np.isfinite(y).all(axis=2), extent, np.stack(bb),
                  np.array(ss), np.zeros(len(y)), np.array(ff))


def fit_predict(data, spec, parts):
    positions, extent, training_targets = learning_view(data, spec)
    if spec['encoder'] == 'template_qsf':
        assert list(data['templates'][spec['split']]['fit_ids']) == parts['fit']
    inner = bundle(data, spec, training_targets, parts['fit'], positions, extent)
    final = bundle(data, spec, training_targets, parts['train'], positions, extent)
    grid, chosen, models, diagnostics = fit(inner, final, parts['fit'], parts['validation'],
                                           parts['train'], spec['count'], 'tanh3')
    selected = robust_choice(chosen)
    if spec['task'] == 'unseen':
        evaluation = bundle(data, spec, np.array(spec['targets']), parts['train'], positions, extent)
        assert not np.isfinite(evaluation.y).any()
    else:
        evaluation = final
    model = models[selected['family']]
    xx = evaluation.x[parts['test']]
    raw = model.predict(xx.reshape(-1,xx.shape[-1])).reshape(len(xx),xx.shape[1],6)
    prediction, clipped = reconstruct(raw,evaluation,parts['test'],selected['family'],selected['radius'])
    diagnostics['training_labels'] = training_targets.tolist()
    diagnostics['finite_training_labels_per_subject_min'] = int(inner.finite[parts['fit']].sum(axis=1).min())
    diagnostics['finite_training_labels_per_subject_max'] = int(inner.finite[parts['fit']].sum(axis=1).max())
    return dict(grid=grid, chosen=chosen, selected=selected, models=models, diagnostics=diagnostics,
                final=final, evaluation=evaluation, prediction=prediction, clipped=clipped, raw=raw)


def directory(spec):
    return OUT/spec['cohort']/'cells'/f"s{spec['split']}_m{spec['mask']}_k{spec['count']}_{spec['task']}_{spec['encoder']}"


def independent_check(result, parts):
    selected = result['selected']
    model = result['models'][selected['family']]
    x,y = result['final'].fit_arrays(parts['train'])
    z = transform((x-model.mean)/model.scale,'tanh3')-model.transformed_mean
    design = np.vstack([z, np.sqrt(selected['alpha'])*np.eye(z.shape[1])])
    target = np.vstack([y-model.y_mean, np.zeros((z.shape[1],y.shape[1]))])
    coef = np.linalg.lstsq(design,target,rcond=None)[0]
    np.testing.assert_allclose(coef,model.coef,atol=1e-8,rtol=1e-7)
    error = np.max(np.abs(z@coef+model.y_mean-model.predict(x)))
    assert error < 1e-7, error
    return float(error)


def poison_check(data, spec, parts, result):
    changed = dict(data)
    changed['positions'] = np.array(data['positions'],copy=True)
    hidden = np.array(spec['targets'])
    changed['positions'][:,hidden] = 1e8
    changed['extent'] = np.full_like(data['extent'],1e12)
    other = fit_predict(changed,spec,parts)
    assert other['selected'] == result['selected']
    assert other['grid'] == result['grid']
    for family in FAMILIES:
        for key in ('mean','scale','transformed_mean','y_mean','coef'):
            np.testing.assert_array_equal(getattr(other['models'][family],key),getattr(result['models'][family],key))
    np.testing.assert_array_equal(other['prediction'],result['prediction'])
    return True


def run_cell(spec):
    folder = directory(spec)
    if (folder/'COMPLETE.json').exists():
        assert verify(folder)['spec'] == spec
        return 'resumed'
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Preserve this incomplete cell and retry in a new work directory: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    previous = load_design()
    parts = original.partition_indices(previous,spec['cohort'],spec['split'],DATA['subjects'])
    result = fit_predict(DATA,spec,parts)
    representative = spec['split'] == 1 and spec['mask'] == 1 and spec['count'] == 32
    recovered = 'solver_recovery' in result['diagnostics']
    independent = independent_check(result,parts) if representative or recovered else None
    poison = poison_check(DATA,spec,parts,result) if (representative or recovered) and spec['task'] == 'unseen' else None
    selected = result['selected']
    pd.DataFrame(result['grid']).to_csv(folder/'validation_grid.csv',index=False)
    dump(folder/'selections.json',dict(families=result['chosen'],robust=selected,diagnostics=result['diagnostics']))
    arrays = {}
    for family, model in result['models'].items():
        for key in ('mean','scale','transformed_mean','y_mean','coef'):
            arrays[family+'_'+key] = getattr(model,key)
    np.savez_compressed(folder/'models.npz',**arrays)
    # Only now attach protected truth and the original extent, strictly for scoring.
    evaluation = replace(result['evaluation'],y=DATA['positions'][:,spec['targets']],extent=DATA['extent'],
                         finite=np.isfinite(DATA['positions'][:,spec['targets']]).all(axis=2))
    test = parts['test']
    nr,nm = losses(result['prediction'],evaluation,test)
    assert np.isfinite(nr).all() and evaluation.finite[test].any(axis=1).all()
    rows = []
    for local,i in enumerate(test):
        rows.append(dict(subject_id=DATA['subjects'][i],normalized_rmse=float(nr[local]),
             normalized_mse=float(nm[local]),native_rmse=float(nr[local]*evaluation.extent[i]),
             target_count=int(evaluation.finite[i].sum()),family=selected['family'],
             alpha=selected['alpha'],radius=selected['radius']))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    np.savez_compressed(folder/'predictions.npz',prediction=result['prediction'],raw=result['raw'],
        truth=evaluation.y[test],finite=evaluation.finite[test],extent=evaluation.extent[test],
        center=evaluation.center[test],scale=evaluation.scale[test],clipped=result['clipped'],
        targets=np.array(spec['targets']),sources=np.array(spec['sources']),
        subject_ids=np.array([DATA['subjects'][i] for i in test]))
    finish(folder,spec=spec,seconds=time.monotonic()-start,
           independent_solver_max_abs_error=independent,hidden_coordinate_poison_passed=poison)
    return 'fit'


def run(cohort, workers):
    check_release()
    original.verify(original.OUT/cohort/'data')
    verify(OUT/cohort/'data')
    specs = [s for s in plans(load_design()) if s['cohort'] == cohort]
    failures = []
    with ProcessPoolExecutor(max_workers=workers,initializer=load_data,initargs=(cohort,)) as pool:
        futures = {pool.submit(run_cell,s):s for s in specs}
        for n, future in enumerate(as_completed(futures),1):
            try:
                future.result()
            except Exception as exc:
                failures.append(dict(spec=futures[future],error=repr(exc)))
                print(f'Failed cell: {failures[-1]}',flush=True)
            if n % 20 == 0 or n == len(specs):
                print(f'{cohort}: {n}/{len(specs)} finished',flush=True)
    if failures:
        dump(OUT/cohort/'FAILED.json',failures)
        raise RuntimeError(f'{len(failures)} cells failed; no reduced analysis is produced.')
    print(f'{cohort}: every planned control completed.',flush=True)
