"""Independent sensor-frame IoU, paired table and numerical-path verification."""
from concurrent.futures import ProcessPoolExecutor
import csv
from multiprocessing import get_context

import numpy as np

from scope174 import (NAMES, ORDER, PRIVATE, RESULTS, THRESHOLDS,
                      camera_view, canonical, json_atomic, ordinary_rows, read, sha)
from independent_audit import independent_annotations, sensor_iou, transform_gt
from predict174 import OUTPUT, verify_predictions
from score174 import METRICS


def recompute(direction):
    source,target=direction
    name=f'{source}_to_{target}'
    folder=OUTPUT/name
    seal=read(folder/'SEAL.json')
    previous=PRIVATE/'score'/f'{name}.npz'
    receipt=read(PRIVATE/'SCORING_SEAL.json')
    assert receipt['score_arrays'][name]==sha(previous)
    with np.load(previous,allow_pickle=False) as z:
        expected,order,expected_slots=(z[k] for k in ('ious','ordinal','slots'))
        assert str(z['predictions'])==sha(folder/'SEAL.json')
    rows=ordinary_rows(target)
    n=0
    error=0.
    disagreements=np.zeros(len(THRESHOLDS),np.int64)
    for file,proof in sorted(seal['files'].items()):
        path=folder/file
        assert sha(path)==proof
        with np.load(path,allow_pickle=False) as z:
            indices=z['ordinals']
            slots=z['subject_slots']
            baseline=z['boxes'][:,0]
            forecast=z['boxes'][:,1:]
            chosen=z['choices'][...,0]
            supports=z['support']
            assert tuple(z['arms'])==NAMES
            np.testing.assert_array_equal(baseline,slots[:,0])
            np.testing.assert_array_equal(forecast,slots[np.arange(len(slots))[:,None],chosen])
            np.testing.assert_allclose(supports.sum(-1,dtype=np.float64),1.,rtol=0.,atol=.003)
            np.testing.assert_array_equal(indices,order[n:n+len(indices)])
        for k,index in enumerate(indices):
            r=rows[int(index)]
            description,raw=independent_annotations(r['metapath'])[r['object_index']]
            assert canonical(description)==r['caption']
            camera=camera_view(r)
            y=transform_gt(raw,target,camera)
            b=sensor_iou(slots[k].astype(float),y,target,camera)
            decoded=b[np.r_[0,chosen[k]]]
            error=max(error,float(abs(decoded-expected[n]).max()),float(abs(b-expected_slots[n]).max()))
            for i,t in enumerate(THRESHOLDS):
                disagreements[i]+=int(((decoded>t)!=(expected[n]>t)).sum())
            n+=1
    assert n==len(rows)==len(expected)
    assert error<1e-7 and not disagreements.any()
    print('INDEPENDENT_SENSOR_IOU',name,n,error,flush=True)
    return name,dict(n=n,all_four_proposals_audited=True,max_abs_error=error,
                     threshold_disagreements=disagreements.tolist())


def reopen(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def verify_tables():
    receipt=read(PRIVATE/'SCORING_SEAL.json')
    for k,h in receipt['public_tables'].items():
        assert sha(RESULTS/k)==h
    endpoints=reopen(RESULTS/'endpoints.csv')
    deltas=reopen(RESULTS/'paired_deltas.csv')
    ns=0
    all_losses=np.zeros(3,np.int64)
    by_cell=[]
    for s,t in ORDER:
        name=f'{s}_to_{t}'
        ledger=reopen(RESULTS/f'paired_{name}.csv')
        a=np.array([[float(r[k]) for k in NAMES] for r in ledger])
        assert len(a)==len(ordinary_rows(t))
        ns+=len(a)
        cube=np.stack((a,a>.25,a>.50),axis=-1).astype(float)
        levels={'all':np.ones(len(a),bool),
                'has_reference':np.array([r['reference_mention']=='1' for r in ledger]),
                'direct_left_right':np.array([r['direct_left_right']=='1' for r in ledger])}
        levels['no_reference']=~levels['has_reference']
        for key,mask in levels.items():
            if not mask.any():
                continue
            group=[r for r in endpoints if (r['scope'],r['cohort'])==(name,key)]
            assert [r['arm'] for r in group]==list(NAMES)
            recorded=np.array([[float(r[k]) for k in METRICS] for r in group])
            np.testing.assert_allclose(cube[mask].mean(0)*100,recorded,rtol=0.,atol=1e-10)
        by_cell.append(cube.mean(0)*100)
        all_losses+=((cube[:,-1]-cube[:,0])< -1e-10).sum(0)
    macro=np.mean(by_cell,axis=0)
    expected=[r for r in endpoints if (r['scope'],r['cohort'])==('six_equal','all')]
    np.testing.assert_allclose(macro,[[float(r[k]) for k in METRICS] for r in expected],rtol=0.,atol=1e-10)
    final=next(r for r in deltas if (r['scope'],r['cohort'],r['contrast'])==('six_equal','all','full_minus_R158'))
    np.testing.assert_array_equal(all_losses,[int(final[k+'_loss']) for k in METRICS])
    assert ns==18620
    return dict(n=ns,full_vs_R158_negative_rows=dict(zip(METRICS,all_losses.tolist(),strict=True)))


def main():
    verify_predictions()
    receipt=read(PRIVATE/'SCORING_SEAL.json')
    assert receipt['prediction']==sha(OUTPUT/'PREDICTION_SEAL.json')
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as workers:
        out=dict(workers.map(recompute,ORDER))
    table=verify_tables()
    path=RESULTS/'independent_audit.json'
    json_atomic(path,dict(n=sum(a['n'] for a in out.values()),directions=out,tables=table,
                          scoring=sha(PRIVATE/'SCORING_SEAL.json'),FINAL_access=False))
    print('COMPLETE_INDEPENDENT_AUDIT',table['n'],flush=True)


if __name__=='__main__':
    main()
