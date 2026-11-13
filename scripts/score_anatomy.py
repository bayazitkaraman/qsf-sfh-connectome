"""Score frozen test predictions; no model fitting or outcome-based selection."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from qsf_sfh.data import ROOT, WORK, check_release

PRIMARY = WORK/'consolidation'
CONTROL = WORK/'controls'
OUT = WORK/'anatomy'
ENC = ('qsf', 'heat_raw18', 'heat_norm18')
FAMILIES = ('ordinary', 'projection', 'residual', 'bounded_residual')
KEY = ['cohort','task','count','model']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def packaged(member):
    return pd.read_csv(ROOT/'results'/member)


def checked(folder, manifest, name):
    p = folder/name
    assert digest(p) == manifest['artifacts'][name], p
    return p


def nearest_other(coords):
    finite = np.isfinite(coords).all(axis=-1)
    dist = np.linalg.norm(coords[:,None]-coords[None,:], axis=2)
    dist[~np.isfinite(dist)] = np.inf
    np.fill_diagonal(dist,np.inf)
    answer = dist.min(axis=1)
    assert np.isfinite(answer[finite]).all()
    return answer


def specs(cohort, split, mask, k, choices, bounds):
    for encoder in ENC:
        for track in ('random','selected'):
            policy = 'random' if track=='random' else choices.loc[(cohort,split,mask,k,encoder),'policy']
            yield 'known', track+'_'+encoder, PRIMARY/cohort/'cells'/f's{split}_m{mask}_k{k}_{policy}_{encoder}_individual_tanh3', False, False
    yield 'known','template_qsf',PRIMARY/cohort/'cells'/f's{split}_m{mask}_k{k}_random_qsf_template_tanh3',False,False
    if not bounds:
        yield 'known','atlas_aligned',PRIMARY/cohort/'cells'/f's{split}_m{mask}_k{k}_random_qsf_individual_tanh3',False,True
    for task, encoders in [('known',['shortest_path']),('unseen',[*ENC,'template_qsf','shortest_path'])]:
        for encoder in encoders:
            yield task, encoder, CONTROL/cohort/'cells'/f's{split}_m{mask}_k{k}_{task}_{encoder}',True,False


def expected(summary, controls, cohort, task, k, model):
    if task=='unseen' or model=='shortest_path':
        return controls.loc[(cohort,task,k,model),'mean_nrmse']
    if model in ('atlas_aligned','template_qsf'):
        enc,track = 'qsf',{'atlas_aligned':'atlas_random_aligned','template_qsf':'template_random_robust'}[model]
    else:
        policy,enc = model.split('_',1)
        track = policy+'_robust'
    return summary.loc[(cohort,k,enc,track),'equal_participant_nrmse']


def run(phase):
    check_release()
    OUT.mkdir(exist_ok=True, parents=True)
    bounds = phase=='bounds'
    choices = packaged('selected_choices.csv').set_index(['cohort','split_id','mask_id','source_count','encoder'])
    base = packaged('track_summary.csv').set_index(['cohort','source_count','encoder','track'])
    controls = packaged('controls_summary.csv').set_index(['cohort','task','count','encoder'])
    rows = []
    for cohort in ('hcp83','oasis'):
        folder = PRIMARY/cohort/'data'
        meta = json.loads((folder/'COMPLETE.json').read_text())
        pos = np.load(checked(folder,meta,'positions.npy'))
        people = json.loads(checked(folder,meta,'subjects.json').read_text())
        index = {s:i for i,s in enumerate(people)}
        spacing = np.stack([nearest_other(x) for x in pos]) if not bounds else None
        for split in range(1,6):
            for mask in range(1,4):
                for k in (8,16,32):
                    alignment = None
                    for task,model,cell,ctl,atlas in specs(cohort,split,mask,k,choices,bounds):
                        m = json.loads((cell/'COMPLETE.json').read_text())
                        sel = json.loads(checked(cell,m,'selections.json').read_text())['robust']
                        with np.load(checked(cell,m,'predictions.npz')) as p:
                            subject = p['subject_ids']
                            ids = np.array([index[str(s)] for s in subject])
                            targets, finite, truth, extent = p['targets'],p['finite'],p['truth'],p['extent']
                            assert not set(p['sources']) & set(targets)
                            np.testing.assert_allclose(pos[ids][:,targets],truth,equal_nan=True)
                            np.testing.assert_array_equal(finite,np.isfinite(truth).all(axis=2))
                            assert finite.any(axis=1).all()
                            identity = (tuple(subject),tuple(targets))
                            if alignment is None: alignment = identity
                            assert identity==alignment
                            if bounds:
                                radius = sel['radius']*p['scale']
                                center = p['center']
                                outside = (np.linalg.norm(truth-center,axis=-1)>radius[:,None]) & finite
                                result = dict(radius_extent=radius/extent, outside_fraction=outside.sum(axis=1)/finite.sum(axis=1),
                                              outside=outside.sum(axis=1))
                            else:
                                if atlas:
                                    with np.load(checked(cell,m,'atlas_predictions.npz')) as a:
                                        np.testing.assert_allclose(a['truth'],truth,equal_nan=True)
                                        pred=a['prediction'][1]
                                else:
                                    pred = p['prediction'] if ctl else p['prediction'][FAMILIES.index(sel['family'])]
                                assert np.isfinite(pred[finite]).all()
                                e = np.linalg.norm(pred-truth,axis=-1)
                                d = spacing[ids][:,targets]
                                n = finite.sum(axis=1)
                                rmse = np.sqrt(np.where(finite,e**2,0).sum(axis=1)/n)
                                result = dict(nrmse=rmse/extent, native_rmse=rmse,
                                              within_spacing=((e<d)&finite).sum(axis=1)/n,
                                              within_half=((e<d/2)&finite).sum(axis=1)/n,
                                              zero_spacing=((d==0)&finite).sum(axis=1))
                            rows.append(pd.DataFrame(dict(cohort=cohort,task=task,count=k,model=model,
                                        split=split,mask=mask,subject=subject,n=finite.sum(axis=1),**result)))
            print(phase,cohort,'split',split,'complete',flush=True)
    records=pd.concat(rows,ignore_index=True)
    metrics = ['radius_extent','outside_fraction'] if bounds else ['nrmse','native_rmse','within_spacing','within_half']
    persons=records.groupby(KEY+['subject'])[metrics].mean()
    means=persons.groupby(KEY).mean()
    means['people']=persons.groupby(KEY).size()
    means['target_records']=records.groupby(KEY).n.sum()
    if bounds:
        means['pooled_outside']=records.groupby(KEY).outside.sum()
        means['radius_extent_median_person']=persons.groupby(KEY).radius_extent.median()
        means['radius_extent_p05_person']=persons.groupby(KEY).radius_extent.quantile(.05)
        means['radius_extent_p95_person']=persons.groupby(KEY).radius_extent.quantile(.95)
    else:
        means['zero_spacing_target_records']=records.groupby(KEY).zero_spacing.sum()
        for key,value in means.nrmse.items():
            np.testing.assert_allclose(value,expected(base,controls,*key),rtol=1e-9,atol=1e-10,err_msg=str(key))
        paired=[]
        for cohort in ('hcp83','oasis'):
            for k in (8,16,32):
                for task,x,y in [('known','random_qsf','random_heat_raw18'),('known','random_qsf','random_heat_norm18'),
                                 ('known','selected_qsf','random_qsf'),('known','random_qsf','template_qsf'),
                                 ('known','random_qsf','atlas_aligned'),('unseen','qsf','heat_raw18'),
                                 ('unseen','qsf','heat_norm18'),('unseen','qsf','template_qsf')]:
                    a,b=persons.loc[(cohort,task,k,x)],persons.loc[(cohort,task,k,y)]
                    assert a.index.equals(b.index)
                    for metric in ['nrmse','within_spacing','within_half']:
                        v=(a[metric]-b[metric]).to_numpy()
                        rng=np.random.default_rng(20261110)
                        boot=np.concatenate([v[rng.integers(0,len(v),(100,len(v)))].mean(axis=1) for _ in range(50)])
                        paired.append(dict(cohort=cohort,task=task,count=k,left=x,right=y,metric=metric,
                                           difference=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975)))
        pd.DataFrame(paired).to_csv(OUT/'practical_paired.csv',index=False)
    means.to_csv(OUT/f'{phase}.csv')
    persons.to_csv(OUT/f'{phase}_participant.csv.gz')
    print('PASS',phase,len(means),'rows',flush=True)


if __name__=='__main__':
    assert sys.argv[1] in ('practical','bounds')
    run(sys.argv[1])
