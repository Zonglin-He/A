"""Score all sealed proposals and paired outcomes under the ordinary target protocol."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from multiprocessing import get_context

import numpy as np

from scope175 import (BASELINES,CELLS,N,NAMES,ORDER,PLATFORMS,PRIVATE,R165_DATA,
                      RELATIONS,RESULTS,frame,json_atomic,npz_atomic,ordinary_rows,read,recording,sha,state)
from predict175 import OUTPUT,PRED_ARMS,verify
from score import labels,local_points
from batch_yaw_iou import many_iou

METRICS=('mIoU','Acc25','Acc50')
COMPARISONS=tuple(('full',m) for m in NAMES if m!='full')
COMPARISONS+=(('pair_only','self_geometry'),('without_dual','self_geometry'))


def write_csv(p,rows):
    assert rows
    t=p.with_suffix('.tmp')
    with t.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    t.replace(p)


def original(route):
    folder=R165_DATA/route
    receipt=read(folder/'BBF_CUDA_RECEIPT.json')
    p=folder/'MATCHED_BOXES_CUDA_PRIVATE.npz'
    assert sha(p)==receipt['output_sha256'] and sha(folder/'SEAL.json')==receipt['direction_forward_seal']
    with np.load(p,allow_pickle=False) as z:
        idx=[list(z['arms']).index(k) for k in BASELINES]
        order=z['ordinals']
        b=z['boxes'][:,idx]
    with np.load(folder/'IOU_AUDIT_PRIVATE.npz',allow_pickle=False) as z:
        assert str(z['input_sha256'])==receipt['output_sha256']
        np.testing.assert_array_equal(order,z['ordinals'])
        iou=z['values'][:,idx]
    np.testing.assert_array_equal(order,np.arange(receipt['n']))
    return b,iou


def score_direction(direction):
    source,target=direction
    route=f'{source}_to_{target}'
    adjacent=ordinary_rows(target)
    folder=OUTPUT/route
    seal=read(folder/'SEAL.json')
    original_boxes,original_iou=original(route)
    record=defaultdict(list)
    for name,digest in sorted(seal['files'].items()):
        p=folder/name
        assert sha(p)==digest
        with np.load(p,allow_pickle=False) as z:
            d=dict(z)
        assert tuple(d['arms'])==PRED_ARMS
        for j,i in enumerate(d['ordinals']):
            row=adjacent[int(i)]
            y,_=labels(row)
            proposals=d['slot_boxes'][j,0]
            q=many_iou(proposals,y)
            choices=d['choice'][j,:,0]
            np.testing.assert_array_equal(d['boxes'][j,:,0],proposals[choices])
            a=many_iou(original_boxes[i],y)
            np.testing.assert_allclose(a,original_iou[i],atol=2e-12,rtol=0.)
            mask=local_points(d['xyz'][j],y)
            p=d['support'][j,:,0].astype(float)
            support=(p*mask).sum(-1)/p.sum(-1)
            observed=dict(ordinal=int(i),ious=np.r_[a,q[choices]],slots=q,
                          choices=choices,relation=d['relation'][j],support=support,
                          field=d['field'][j],frame=frame(row),recording=recording(row))
            for k,v in observed.items():
                record[k].append(v)
    p=PRIVATE/'score'/f'{route}.npz'
    np.testing.assert_array_equal(record['ordinal'],np.arange(seal['n']))
    npz_atomic(p,**{k:np.asarray(v) for k,v in record.items()},prediction=sha(folder/'SEAL.json'))
    print('R175_SCORED',route,seal['n'],flush=True)
    return route,sha(p)


def cells(x):
    return np.stack((x,x>.25,x>.50),axis=-1).astype(float)


def masks(kind):
    return {'all':np.ones(len(kind),bool),'has_relation':kind!=0,
            'explicit_left_right':(kind>0)&(kind<=2),'no_relation':kind==0}


def tables(results):
    outcomes,paired,supports,fields=[],[],[],[]
    for cohort in ('all','has_relation','explicit_left_right','no_relation'):
        valid={k:masks(x['relation'])[cohort] for k,x in results.items()}
        for scope in (*results,'six_equal','pooled'):
            use=list(results) if scope in ('six_equal','pooled') else [scope]
            views=[{k:x[valid[d]] for k,x in results[d].items() if k!='prediction'}
                   for d in use if valid[d].any()]
            if not views:
                continue
            values=np.concatenate([v['ious'] for v in views])
            y=cells(values)
            av=(np.mean([cells(v['ious']).mean(0) for v in views],axis=0)
                if scope=='six_equal' else y.mean(0))*100.
            frames=np.concatenate([v['frame'] for v in views])
            recs=np.concatenate([v['recording'] for v in views])
            header=dict(scope=scope,cohort=cohort,n=len(values),directions=len(views),
                        physical_frames=len(set(frames)),recordings=len(set(recs)))
            for j,key in enumerate(NAMES):
                outcomes.append(dict(**header,arm=key,**dict(zip(METRICS,av[j],strict=True))))
            slot_ious=np.concatenate([v['slots'].max(1) for v in views])
            for j,key in enumerate(PRED_ARMS):
                mass=np.concatenate([v['support'][:,j] for v in views])
                bias=np.concatenate([v['field'][:,j] for v in views])
                chosen=values[:,len(BASELINES)+j]
                supports.append(dict(**header,arm=key,mean_subject_box_support=float(mass.mean()),
                                     reaches25=int((slot_ious>.25).sum()),
                                     reaches50=int((slot_ious>.50).sum()),
                                     missed25=int(((slot_ious>.25)&(chosen<=.25)).sum()),
                                     missed50=int(((slot_ious>.50)&(chosen<=.50)).sum())))
                fields.append(dict(**header,arm=key,mean_field=float(bias.mean()),
                                   n_nonzero=int((bias.max(1)>1e-8).sum())))
            for a,b in COMPARISONS:
                i,j=NAMES.index(a),NAMES.index(b)
                change=y[:,i]-y[:,j]
                flags={k+'_'+name:int((c>d).sum()) for k,c in zip(METRICS,change.T,strict=True)
                       for name,d in (('gain',1e-10),)}
                flags.update({k+'_loss':int((c< -1e-10).sum()) for k,c in zip(METRICS,change.T,strict=True)})
                flags['iou_equal']=int((abs(change[:,0])<=1e-10).sum())
                paired.append(dict(**header,arm=a,comparison=b,**flags,
                                   **{k+'_pp':v for k,v in zip(METRICS,av[i]-av[j],strict=True)}))
    return {'endpoints.csv':outcomes,'paired_deltas.csv':paired,'support_and_coverage.csv':supports,
            'field_activity.csv':fields}


def intervals(data):
    rng=np.random.default_rng(175)
    loops=4000
    means,counts={},{}
    for target in PLATFORMS:
        keys=[f'{s}_to_{target}' for s in PLATFORMS if s!=target]
        studies=[data[r] for r in keys]
        names,c=np.unique(studies[0]['recording'],return_inverse=True)
        np.testing.assert_array_equal(studies[0]['recording'],studies[1]['recording'])
        assert len(names)>1
        counts[target]=len(names)
        draw=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=loops)
        den=(draw@np.bincount(c))[:,None,None]
        for route,v in zip(keys,studies,strict=True):
            reduction=np.zeros((len(names),len(NAMES),len(METRICS)))
            np.add.at(reduction,c,cells(v['ious']))
            means[route]=100.*np.einsum('ij,jad->iad',draw,reduction)/den
    z=np.mean(list(means.values()),axis=0)
    result={}
    for a,b in COMPARISONS:
        delta=z[:,NAMES.index(a)]-z[:,NAMES.index(b)]
        result[f'{a}_minus_{b}']=dict(zip(METRICS,np.quantile(delta,(.025,.975),axis=0).T.tolist(),strict=True))
    return dict(sample_unit='recording within target; joint draws for two sources',n=loops,
                n_recordings=counts,descriptive_intervals_pp=result)


def main():
    verify()
    state('post_seal_scoring','Score frozen endpoints, within-recording deltas and box support.')
    (PRIVATE/'score').mkdir(exist_ok=True,mode=0o700)
    RESULTS.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        scored=dict(pool.map(score_direction,ORDER))
    data={}
    checks={}
    listing=np.random.default_rng(1752)
    for route,h in scored.items():
        path=PRIVATE/'score'/f'{route}.npz'
        with np.load(path,allow_pickle=False) as z:
            data[route]=dict(z)
        rows=data[route]
        table=[]
        for i in listing.permutation(len(rows['ious'])):
            table.append(dict(relation=RELATIONS[rows['relation'][i]],best_slot_iou=rows['slots'][i].max(),
                              **dict(zip(NAMES,rows['ious'][i],strict=True))))
        path=RESULTS/f'paired_{route}.csv'
        write_csv(path,table)
        checks[path.name]=sha(path)
    for name,rows in tables(data).items():
        path=RESULTS/name
        write_csv(path,rows)
        checks[name]=sha(path)
    interval=RESULTS/'recording_intervals.json'
    json_atomic(interval,intervals(data))
    checks[interval.name]=sha(interval)
    assert sum(len(q['ious']) for q in data.values())==N
    json_atomic(PRIVATE/'SCORING_SEAL.json',dict(n=N,predictions=sha(OUTPUT/'PREDICTION_SEAL.json'),
                      result_hashes=checks,scored=scored,FINAL_access=False))
    print('R175_SCORE_SEALED',N,flush=True)


if __name__=='__main__':
    main()
