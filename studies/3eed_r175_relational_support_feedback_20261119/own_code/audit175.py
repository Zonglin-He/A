"""Independent sensor-coordinates IoU and full paired scalar audit."""
from concurrent.futures import ProcessPoolExecutor
import csv
from multiprocessing import get_context

import numpy as np

from scope175 import (BASELINES,N,NAMES,OLD_PRIVATE,ORDER,PLATFORMS,PRIVATE,RELATIONS,RESULTS,
                      camera_view,frame,json_atomic,ordinary_rows,read,recording,sha)
# Register the local evidence module before importing the legacy audit dependencies.
from predict175 import OUTPUT,PRED_ARMS,verify
from score175 import original
from independent_audit import canonical,independent_annotations,sensor_iou,transform_gt
from batch_yaw_iou import many_iou


def geometric(direction):
    source,target=direction
    route=f'{source}_to_{target}'
    folder=OUTPUT/route
    sealed=read(folder/'SEAL.json')
    score=PRIVATE/'score'/f'{route}.npz'
    receipt=read(PRIVATE/'SCORING_SEAL.json')
    assert sha(score)==receipt['scored'][route]
    with np.load(score,allow_pickle=False) as z:
        expected=z['ious']
        slots=z['slots']
        assert str(z['prediction'])==sha(folder/'SEAL.json')
    historical,h_ious=original(route)
    error=0.
    threshold=np.zeros(2,int)
    rows=ordinary_rows(target)
    offset=0
    for name,digest in sorted(sealed['files'].items()):
        f=folder/name
        assert sha(f)==digest
        with np.load(f,allow_pickle=False) as z:
            ordinals=z['ordinals']
            a=z['slot_boxes'][:,0]
            selected=z['choice'][:,:,0]
            raw=z['boxes'][:,:,0]
            assert tuple(z['arms'])==PRED_ARMS
            np.testing.assert_array_equal(raw,a[np.arange(len(a))[:,None],selected])
            np.testing.assert_allclose(z['support'].sum(-1,dtype=np.float64),1.,atol=.003,rtol=0.)
        np.testing.assert_array_equal(ordinals,np.arange(offset,offset+len(a)))
        for j,i in enumerate(ordinals):
            row=rows[int(i)]
            caption,truth_raw=independent_annotations(row['metapath'])[row['object_index']]
            assert canonical(caption)==row['caption']
            cam=camera_view(row)
            truth=transform_gt(truth_raw,target,cam)
            alt=sensor_iou(np.concatenate((historical[i],a[j]),0),truth,target,cam)
            p=alt[:len(BASELINES)]
            q=alt[len(BASELINES):]
            chosen=np.r_[p,q[selected[j]]]
            error=max(error,float(abs(alt-np.r_[h_ious[i],slots[i]]).max()),
                      float(abs(chosen-expected[i]).max()))
            for m,t in enumerate((.25,.5)):
                threshold[m]+=np.count_nonzero((chosen>t)!=(expected[i]>t))
        offset+=len(ordinals)
    assert offset==len(expected) and error<1e-7 and not threshold.any()
    print('R175_INDEPENDENT_IOU',route,offset,error,flush=True)
    return route,dict(n=offset,max_abs_error=error,threshold_disagreements=threshold.tolist())


def csv_rows(name):
    with (RESULTS/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def scalar_audit():
    receipt=read(PRIVATE/'SCORING_SEAL.json')
    for p,h in receipt['result_hashes'].items():
        assert sha(RESULTS/p)==h
    old=csv_rows('endpoints.csv')
    table={(r['scope'],r['cohort'],r['arm']):r for r in old}
    metrics=('mIoU','Acc25','Acc50')
    six=[]
    total=0
    for source,target in ORDER:
        route=f'{source}_to_{target}'
        found=csv_rows(f'paired_{route}.csv')
        iou=np.array([[float(q[a]) for a in NAMES] for q in found])
        flag=np.array([q['relation'] for q in found])
        assert len(iou)==len(ordinary_rows(target))
        total+=len(iou)
        v=np.stack((iou,iou>.25,iou>.50),-1).astype(float)
        groups={'all':np.ones(len(iou),bool),'has_relation':flag!='none',
                'explicit_left_right':np.isin(flag,['left','right']),'no_relation':flag=='none'}
        for g,b in groups.items():
            if b.any():
                actual=np.array([[float(table[route,g,k][m]) for m in metrics] for k in NAMES])
                np.testing.assert_allclose(actual,100*v[b].mean(0),atol=1e-10,rtol=0.)
        six.append(v.mean(0)*100.)
    actual=np.array([[float(table['six_equal','all',k][m]) for m in metrics] for k in NAMES])
    np.testing.assert_allclose(actual,np.mean(six,0),atol=1e-10,rtol=0.)
    assert total==N
    return total


def audit_source_rosters():
    from source175 import inventory_for
    result={}
    for source in PLATFORMS:
        history={}
        split_info={}
        for split in ('train','monitor'):
            old=read(OLD_PRIVATE/source/f'{split}_PRIVATE.json')
            new=read(PRIVATE/source/f'{split}_PRIVATE.json')
            natural=[r for r in new if r['origin']!='synthetic_labelled_pair']
            old_nat=[r for r in old if not r['known_reference']]
            assert len(old_nat)==len(natural)
            for a,b in zip(old_nat,natural,strict=True):
                assert all(a[k]==b[k] for k in ('caption','metapath','pointpath','seed','platform','suffix'))
                assert a['boxes'][0]==b['boxes'][0]
            seen=set()
            positive=0
            for row in new:
                assert row['platform']==source
                if not row['known_reference']:
                    continue
                truth=np.asarray(row['boxes'])
                names,boxes,valid,low,high,centres=inventory_for(row)
                available=np.stack([many_iou(boxes,g) for g in truth])
                i,j=available.argmax(-1)
                assert i!=j and available[0,i]>.95 and available[1,j]>.95
                assert valid[i] and valid[j]
                predicate=RELATIONS[row['relation']]
                assert predicate!='none'
                if row['origin']=='synthetic_labelled_pair':
                    f=frame(row)
                    assert f not in seen
                    seen.add(f)
                    foils=valid & (names==names[i]) & (np.arange(len(boxes))!=j)
                    assert foils.sum()>=2
                    if predicate=='near':
                        d=np.linalg.norm(boxes[foils,:3]-boxes[j,:3],axis=1)
                        order=np.argsort(d)
                        assert np.flatnonzero(foils)[order[0]]==i and d[order[1]]-d[order[0]]>=.75
                if predicate!='near':
                    k={'left':0,'right':0,'above':1,'below':1,'closer':2,'farther':2}[predicate]
                    upper,lower=(i,j) if predicate in ('left','above','closer') else (j,i)
                    assert low[lower,k]-high[upper,k]>(.012 if k<2 else .35)
                positive+=1
            history[split]={recording(r) for r in new}
            split_info[split]=dict(natural=len(natural),positive_identity_pairs=positive,
                                   synthetic_frames=len(seen),recordings=len(history[split]))
        assert history['train'].isdisjoint(history['monitor'])
        result[source]=split_info
    return result


def main():
    verify()
    r=read(PRIVATE/'SCORING_SEAL.json')
    assert r['predictions']==sha(OUTPUT/'PREDICTION_SEAL.json')
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        records=dict(pool.map(geometric,ORDER))
    total=scalar_audit()
    assert sum(r['n'] for r in records.values())==total
    sources=audit_source_rosters()
    json_atomic(RESULTS/'independent_audit.json',dict(n=total,directions=records,source_rosters=sources,FINAL_access=False,
                     scoring=sha(PRIVATE/'SCORING_SEAL.json'),prediction=sha(OUTPUT/'PREDICTION_SEAL.json')))
    print('R175_COMPLETE_AUDIT',total,flush=True)


if __name__=='__main__':
    main()
