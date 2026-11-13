"""Matched-cohort random-reference sensitivity using the submitted numerical code."""
import os
for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[var]='1'
import sys
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import argparse
import numpy as np
import pandas as pd

RELEASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RELEASE/'src'))
from qsf_sfh import model, core, control_solver
from qsf_sfh.data import check_release, load_design, digest, WORK
from qsf_sfh.benchmark import qsf_stack
from brain_quantum.analysis_suite import position_extent, compute_walk_stack
from brain_quantum.dense_heat_capacity_control import dense_heat_kernels
from brain_quantum.methodology_controls import balanced_node_folds
from brain_quantum.qrc_hierarchy_suite import nested_random_ranking
from brain_quantum.qrc_connectome import parse_graphml,transform_weights,normalized_laplacian

CACHE=WORK/'hcp129'
ENC=model.ENCODERS
DATA={}


def dump(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare(directory):
    check_release()
    CACHE.mkdir(exist_ok=True, parents=True)
    (CACHE/'cells').mkdir(exist_ok=True)
    design=load_design()['cohorts']['hcp83']
    expected=json.loads((RELEASE/'configs/hcp129_inputs.json').read_text())['graph_sha256']
    if not directory.is_dir():
        raise FileNotFoundError('Supply the extracted HCP-129 GraphML directory.')
    needed=set(expected)
    paths={}
    for path in sorted(directory.rglob('*.graphml')):
        key=digest(path)
        if key in needed:
            paths.setdefault(key,path)
    if len(paths)!=len(expected) or len(paths)!=1062:
        raise ValueError('Missing or duplicate required HCP-129 inputs; no reduced cohort is substituted.')
    n=len(expected)
    coords=np.empty((n,129,3))
    active=np.empty((n,129),bool)
    q=np.lib.format.open_memmap(CACHE/'qsf.npy',mode='w+',dtype=np.float32,shape=(n,18,129,129))
    h=np.lib.format.open_memmap(CACHE/'heat.npy',mode='w+',dtype=np.float32,shape=q.shape)
    for i,key in enumerate(expected):
        g=parse_graphml(paths[key])
        sig=(g.node_ids,g.node_names,g.hemispheres)
        if i==0: reference=sig
        assert sig==reference
        w=transform_weights(g.adjacency,'log1p').astype('float32')
        active[i]=w.sum(axis=1)>0
        coords[i]=g.positions
        q[i]=qsf_stack(compute_walk_stack(normalized_laplacian(w),[.25,.5,1.,2.,4.,8.]))
        h[i]=dense_heat_kernels(SimpleNamespace(weights=w),np.geomspace(.25,8,18).tolist())
        if (i+1)%250==0: print('Hash-verified graphs prepared',i+1,flush=True)
    q.flush()
    h.flush()
    np.save(CACHE/'positions.npy',coords)
    np.save(CACHE/'extent.npy',np.array([position_extent(x) for x in coords]))
    masks=balanced_node_folds(np.arange(129),reference[2],3,np.random.default_rng(18051))
    assert sorted(np.concatenate(masks))==list(range(129))
    jobs=[]
    for split in range(1,6):
        parts=design['partitions'][str(split)]
        assert set(parts['train'])==set(parts['fit'])|set(parts['validation'])
        assert sorted(parts['train']+parts['test']+parts['calibration'])==list(range(n))
        fit=parts['fit']
        eligible=active[fit].any(axis=0)&np.isfinite(coords[fit]).all(axis=2).any(axis=0)
        for m,targets in enumerate(masks,1):
            candidates=np.setdiff1d(np.flatnonzero(eligible),targets)
            ranking=nested_random_ranking(reference[2],np.random.default_rng(18071+100*split+m),candidates)
            sources=[int(i) for i in ranking if i in candidates]
            assert len(set(sources))>=32 and not set(sources)&set(targets)
            for k in (8,16,32):
                for enc in ENC:
                    jobs.append(dict(split=split,mask=m,count=k,encoder=enc,targets=targets.tolist(),
                                     sources=sources[:k],parts=parts))
    dump(CACHE/'design.json',dict(graph_sha256=expected,jobs=jobs,masks=[x.tolist() for x in masks],
         cache_sha256={s:digest(CACHE/(s+'.npy')) for s in ('positions','extent','qsf','heat')},
         release_protocol_sha256=digest(RELEASE/'configs/protocol.txt')))
    print('HCP-129 design locked:',len(jobs),'cells',flush=True)


def init():
    global DATA
    DATA={s:np.load(CACHE/(s+'.npy'),mmap_mode='r') for s in ('positions','extent','qsf','heat')}


def cell(job):
    s,m,k,enc=(job[n] for n in ('split','mask','count','encoder'))
    stem=f's{s}_m{m}_k{k}_{enc}'
    folder=CACHE/'cells'
    src,tgt=np.array(job['sources']),np.array(job['targets'])
    parts=job['parts']
    inner=model.make_bundle(DATA,src,tgt,enc,parts['fit'])
    final=model.make_bundle(DATA,src,tgt,enc,parts['train'])
    grid,choices,models,diagnostics=control_solver.fit(inner,final,parts['fit'],parts['validation'],parts['train'],k,'tanh3')
    selected=model.robust_choice(choices)
    family=selected['family']
    test=parts['test']
    xx=final.x[test]
    raw=models[family].predict(xx.reshape(-1,xx.shape[-1])).reshape(len(test),len(tgt),6)
    pred,_=core.reconstruct(raw,final,test,family,selected['radius'])
    assert np.isfinite(pred).all()
    nr,_=core.losses(pred,final,test)
    np.savez_compressed(folder/(stem+'.npz'),pred=pred,test=np.array(test),targets=tgt,sources=src,nrmse=nr)
    dump(folder/(stem+'.json'),dict(selected=selected,grid=grid,diagnostics=diagnostics,job=job,
         sha256=digest(folder/(stem+'.npz'))))
    return stem


def run(workers):
    design=json.loads((CACHE/'design.json').read_text())
    for key,h in design['cache_sha256'].items():
        assert digest(CACHE/(key+'.npy'))==h
    jobs=design['jobs']
    # Each fit uses participant-by-target arrays and a tall SVD; cap memory use.
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        tasks=[pool.submit(cell,j) for j in jobs]
        for i,f in enumerate(as_completed(tasks),1):
            stem=f.result()
            if i%10==0 or i==len(tasks): print(i,'/',len(tasks),stem,flush=True)
    summarize()


def summarize():
    jobs=json.loads((CACHE/'design.json').read_text())['jobs']
    assert len(jobs)==135
    pos=np.load(CACHE/'positions.npy',mmap_mode='r')
    ext=np.load(CACHE/'extent.npy')
    rows=[]
    for j in jobs:
        stem='s{split}_m{mask}_k{count}_{encoder}'.format(**j)
        p=CACHE/'cells'/(stem+'.npz')
        sel=json.loads(p.with_suffix('.json').read_text())
        assert sel['job']==j and digest(p)==sel['sha256']
        assert model.robust_choice(model.choose_families(sel['grid']))==sel['selected']
        with np.load(p) as a:
            ids=j['parts']['test']
            np.testing.assert_array_equal(a['test'],ids)
            np.testing.assert_array_equal(a['targets'],j['targets'])
            np.testing.assert_array_equal(a['sources'],j['sources'])
            truth=pos[ids][:,j['targets']]
            n=np.sqrt(np.nanmean(np.sum((truth-a['pred'])**2,axis=2),axis=1))/ext[ids]
            np.testing.assert_allclose(n,a['nrmse'],rtol=1e-12,atol=1e-12)
            rows.append(pd.DataFrame(dict(subject=ids,split=j['split'],mask=j['mask'],count=j['count'],encoder=j['encoder'],nrmse=n)))
    allrows=pd.concat(rows,ignore_index=True)
    pivot=allrows.pivot(index=['count','subject','split','mask'],columns='encoder',values='nrmse')
    assert not pivot.isna().any().any()
    means=pivot.groupby(['count','subject']).mean()
    out=[]
    for k in (8,16,32):
        v=means.loc[k]
        assert len(v)==729
        for heat in ('heat_raw18','heat_norm18'):
            dif=(v.qsf-v[heat]).to_numpy()
            rng=np.random.default_rng(20261110)
            boot=np.concatenate([dif[rng.integers(0,len(v),(100,len(v)))].mean(axis=1) for _ in range(50)])
            out.append(dict(count=k,comparator=heat,participants=len(v),qsf=v.qsf.mean(),heat=v[heat].mean(),
                            difference=dif.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),
                            relative_reduction=1-v.qsf.mean()/v[heat].mean(),
                            favorable_split_masks=int(((pivot.qsf-pivot[heat]).loc[k].groupby(['split','mask']).mean()<0).sum())))
    pd.DataFrame(out).to_csv(CACHE/'hcp129.csv',index=False)
    means.to_csv(CACHE/'participants.csv')
    print('PASS 135 verified fits;',len(pivot),'aligned records',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(os.environ.get('QSF_HCP129_DATA',RELEASE/'data/hcp129')))
    parser.add_argument('--stage',choices=('all','prepare','fit','summarize'),default='all')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.workers < 1: parser.error('--workers must be positive')
    check_release()
    if args.stage in ('all','prepare') and not (CACHE/'design.json').exists(): prepare(args.data)
    if args.stage in ('all','fit'): run(args.workers)
    elif args.stage=='summarize': summarize()
