"""Score and report all paired ordinary predictions after the prediction seal."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from multiprocessing import get_context

import numpy as np

from scope174 import (CELLS, NAMES, ORDER, PLATFORMS, PRIVATE, R165_DATA, RESULTS, ROOT,
                      THRESHOLDS, frame, json_atomic, npz_atomic, ordinary_rows, read,
                      recording, sha, state)
from predict174 import OUTPUT, verify_predictions
from score import inventory, labels, local_points, owner_for
from batch_yaw_iou import many_iou

METRICS = ('mIoU','Acc25','Acc50')
CROSS = (0,-1,0,1,0)
SELF = (0,-1,1,0,0)
CONTRASTS = {
    'unary_minus_R158':(-1,1,0,0,0),
    'self_minus_R158':(-1,0,1,0,0),
    'cross_minus_R158':(-1,0,0,1,0),
    'full_minus_R158':(-1,0,0,0,1),
    'full_minus_unary':(0,-1,0,0,1),
    'self_given_no_cross':SELF,
    'self_given_cross':(0,0,0,-1,1),
    'cross_given_no_self':CROSS,
    'cross_given_self':(0,0,-1,0,1),
    'self_main':(0,-.5,.5,-.5,.5),
    'cross_main':(0,-.5,-.5,.5,.5),
    'interaction':(0,1,-1,-1,1)}


def write_csv(path, rows):
    assert rows
    stage = path.with_suffix('.tmp')
    with stage.open('w',encoding='utf-8',newline='') as f:
        out=csv.DictWriter(f,list(rows[0]),lineterminator='\n')
        out.writeheader()
        out.writerows(rows)
    stage.replace(path)


def score_direction(direction):
    source,target = direction
    name=f'{source}_to_{target}'
    folder=OUTPUT/name
    sealed=read(folder/'SEAL.json')
    cases=ordinary_rows(target)
    prior=R165_DATA/name
    receipt=read(prior/'BBF_CUDA_RECEIPT.json')
    with np.load(prior/'IOU_AUDIT_PRIVATE.npz',allow_pickle=False) as z:
        assert str(z['input_sha256'])==receipt['output_sha256']
        reference=z['values'][:,-1]
        np.testing.assert_array_equal(z['ordinals'],np.arange(sealed['n']))
    data=defaultdict(list)
    for fname,digest in sorted(sealed['files'].items()):
        path=folder/fname
        assert sha(path)==digest
        with np.load(path,allow_pickle=False) as z:
            block=dict(z)
        assert tuple(block['arms'])==NAMES
        flags=np.stack((block['reference_mention'],block['direct_left_right']),-1)
        choices=np.c_[np.zeros(len(flags),np.uint8),block['choices'][...,0]]
        for i,ordinal in enumerate(block['ordinals']):
            r=cases[int(ordinal)]
            gt, annotations=labels(r)
            ious=many_iou(block['subject_slots'][i],gt)
            actual=ious[choices[i]]
            np.testing.assert_allclose(reference[ordinal],actual[0],rtol=0.,atol=2e-12)
            objects=inventory(r,gt,annotations)
            xyz=np.unique(block['seeds'][i],axis=0)
            counts=np.stack([local_points(xyz,b) for b in objects])
            ids=np.where(counts.sum(0)==1,counts.argmax(0),len(objects))
            ident=np.asarray([owner_for(b,xyz,ids,len(objects)) for b in block['subject_slots'][i]])
            # At most 2*.75 in unary correction and 2*.5 in reference marginal.
            rho=1.5+int(flags[i,0])
            reachable=np.where(block['anchor_logits'][i]>=-rho,ious,0.)
            vals=dict(ordinal=ordinal, ious=actual, choice=choices[i], slots=ious,
                      full_bound_slot_best=reachable.max(),
                      identity=ident[choices[i]], slot_owner=ident, flags=flags[i],
                      recording=recording(r), frame=frame(r))
            for k,v in vals.items():
                data[k].append(v)
    assert len(data['ious'])==sealed['n']
    info={k:np.asarray(v) for k,v in data.items()}
    np.testing.assert_array_equal(info['ordinal'],np.arange(sealed['n']))
    loc=PRIVATE/'score'/f'{name}.npz'
    npz_atomic(loc,**info,predictions=sha(folder/'SEAL.json'))
    print('SEALED_SCORE',name,len(info['ious']),flush=True)
    return name,sha(loc)


def metrics(x):
    return np.stack((x,x>.25,x>.50),axis=-1).astype(np.float64)


def masks(s):
    r,lr=s['flags'].T
    return {'all':np.ones(len(r),bool),'has_reference':r,'direct_left_right':lr,'no_reference':~r}


def outcomes(ref,alt):
    delta=metrics(alt)-metrics(ref)
    return tuple((int((x>1e-10).sum()),int((x< -1e-10).sum()))
                 for x in delta.T)


def tables(paired):
    names=list(paired)
    endpoints, pairs, effects, ceilings = [], [], [], []
    for cohort in ('all','has_reference','direct_left_right','no_reference'):
        views={}
        for key,s in paired.items():
            m=masks(s)[cohort]
            if m.any():
                fields=('ious','choice','slots','identity','slot_owner','frame','recording','full_bound_slot_best')
                views[key]={k:s[k][m] for k in fields}
        for scope in (*names,'six_equal','row_weighted'):
            group=list(views) if scope in ('six_equal','row_weighted') else [scope] if scope in views else []
            if not group:
                continue
            joined={k:np.concatenate([views[g][k] for g in group]) for k in views[group[0]]}
            mu=(np.mean([metrics(views[g]['ious']).mean(0) for g in group],axis=0)
                if scope=='six_equal' else metrics(joined['ious']).mean(0))*100.
            header=dict(scope=scope,cohort=cohort,n=len(joined['ious']),directions=len(group),
                        recordings=len(set(joined['recording'])),physical_frames=len(set(joined['frame'])))
            oracle=joined['slots'].max(1)
            oracle_mu=(np.mean([metrics(views[g]['slots'].max(1)).mean(0) for g in group],axis=0)
                       if scope=='six_equal' else metrics(oracle).mean(0))*100.
            for j,arm in enumerate(NAMES):
                i=joined['identity'][:,j]
                endpoints.append(dict(**header,arm=arm,**dict(zip(METRICS,mu[j],strict=True)),
                    correct_subject_owner=int((i==0).sum()),wrong_subject_owner=int((i>0).sum()),
                    UNKNOWN_subject_owner=int((i<0).sum()),changed_from_R158=int(np.count_nonzero(joined['choice'][:,j]))))
                y=joined['ious'][:,j]
                ceilings.append(dict(**header,arm=arm,oracle_mIoU=oracle_mu[0],oracle_Acc25=oracle_mu[1],
                    oracle_Acc50=oracle_mu[2],missed_if_slot_available_at25=int(((y<=.25)&(oracle>.25)).sum()),
                    missed_if_slot_available_at50=int(((y<=.50)&(oracle>.50)).sum()),
                    known_other_subject_within_slots_at25=int(((y<=.25)&(oracle>.25)&(i>0)).sum()),
                    known_other_subject_within_slots_at50=int(((y<=.50)&(oracle>.50)&(i>0)).sum()),
                    full_bound_blocks_all_good_slots_at25=int(((oracle>.25)&(joined['full_bound_slot_best']<=.25)).sum()),
                    full_bound_blocks_all_good_slots_at50=int(((oracle>.50)&(joined['full_bound_slot_best']<=.50)).sum())))
            for label,w in CONTRASTS.items():
                score=np.asarray(w)@mu
                effects.append(dict(**header,contrast=label,**dict(zip(METRICS,score,strict=True))))
                if np.count_nonzero(w)!=2:
                    continue
                src,alt=w.index(-1),w.index(1)
                values=joined['ious']
                change=outcomes(values[:,src],values[:,alt])
                effects_counts={f'{metric}_{suffix}':v for metric,pair in zip(METRICS,change,strict=True)
                                for suffix,v in zip(('gain','loss'),pair,strict=True)}
                signed=values[:,alt]-values[:,src]
                pairs.append(dict(**header,contrast=label,after=NAMES[alt],before=NAMES[src],
                    changed_proposal=int((joined['choice'][:,src]!=joined['choice'][:,alt]).sum()),
                    iou_same=int((abs(signed)<=1e-10).sum()),**effects_counts,
                    **{m+'_delta_pp':v for m,v in zip(METRICS,score,strict=True)}))
    return {'endpoints.csv':endpoints,'paired_deltas.csv':pairs,'factorial_effects.csv':effects,
            'candidate_coverage.csv':ceilings}


def clustered(paired, n_boot=4000):
    rng=np.random.default_rng(174)
    counts, means={}, {}
    for target in PLATFORMS:
        received=[f'{s}_to_{target}' for s in PLATFORMS if s!=target]
        candidates=[paired[d] for d in received]
        keys, index=np.unique(candidates[0]['recording'],return_inverse=True)
        np.testing.assert_array_equal(candidates[1]['recording'],candidates[0]['recording'])
        if len(keys)<2:
            return dict(n_boot=0,reason='At least one target contains fewer than two recordings.')
        draw=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=n_boot)
        freq=np.bincount(index).astype(float)
        counts[target]=len(keys)
        for route,section in zip(received,candidates,strict=True):
            data=metrics(section['ious'])
            sample=np.zeros((len(keys),len(NAMES),len(METRICS)))
            np.add.at(sample,index,data)
            means[route]=100*np.einsum('ij,jam->iam',draw,sample)/(draw@freq)[:,None,None]
    z=np.mean(list(means.values()),axis=0)
    limits={}
    for label,w in CONTRASTS.items():
        draws=np.einsum('iam,a->im',z,w)
        interval=np.quantile(draws,(.025,.975),axis=0).T
        limits[label]=dict(zip(METRICS,interval.tolist(),strict=True))
    return dict(n_boot=n_boot,recordings_by_target=counts,interval95_pp=limits,
                unit='recordings resampled within target; both source directions share each resample',
                scope='descriptive sampling spread in the available recordings')


def load_scores():
    paired={}
    for s,t in ORDER:
        p=PRIVATE/'score'/f'{s}_to_{t}.npz'
        with np.load(p,allow_pickle=False) as z:
            paired[p.stem]=dict(z)
    return paired


def main():
    verify_predictions()
    state('post_seal_scoring','All target predictions sealed. Scoring paired endpoints and candidate coverage.')
    RESULTS.mkdir(exist_ok=True)
    (PRIVATE/'score').mkdir(exist_ok=True,mode=0o700)
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as exe:
        hashes=dict(exe.map(score_direction,ORDER))
    paired=load_scores()
    scalars={}
    for name,part in tables(paired).items():
        write_csv(RESULTS/name,part)
        scalars[name]=sha(RESULTS/name)
    rng=np.random.default_rng(1742)
    for name,data in paired.items():
        rows=[]
        order=rng.permutation(len(data['ious']))
        for k in order:
            rows.append(dict(reference_mention=int(data['flags'][k,0]),
                             direct_left_right=int(data['flags'][k,1]),
                             candidate_set_best_iou=data['slots'][k].max(),
                             **dict(zip(NAMES,data['ious'][k],strict=True))))
        filename=f'paired_{name}.csv'
        write_csv(RESULTS/filename,rows)
        scalars[filename]=sha(RESULTS/filename)
    path=RESULTS/'paired_recording_intervals.json'
    json_atomic(path,clustered(paired))
    scalars[path.name]=sha(path)
    receipt=dict(n=sum(len(d['ious']) for d in paired.values()),prediction=sha(OUTPUT/'PREDICTION_SEAL.json'),
                 score_arrays=hashes,public_tables=scalars,ordinary_only=True,FINAL_access=False)
    assert receipt['n']==18620
    json_atomic(PRIVATE/'SCORING_SEAL.json',receipt)
    print('PAIRED_ALL_SCORED',receipt['n'],flush=True)


if __name__=='__main__':
    main()
