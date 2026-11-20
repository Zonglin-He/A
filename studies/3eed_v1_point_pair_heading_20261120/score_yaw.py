"""Open original labels after the common seal; compare all outcomes on matching records."""
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from predict_yaw import ARMS, HERE, PRIVATE, RESULTS, SIZES, STAGES, digest, inputs, read, roster_digest, save
from assess import measure, record_intervals, score_boxes, sensor_truth
from calibrate import csv_file

POOL=HERE.parent/'3eed_k16_utility_loss_20261120'/'run_PRIVATE/scores'
HIST=HERE.parent/'3eed_v1_shared_heading_overlap_20261120'/'results'
METHODS=(*ARMS,'gt_yaw_diagnostic')
PAIRS=[(name,'v1') for name in METHODS[1:]]+[('shared',name) for name in ARMS[2:]]


def anon(message,n):
    return hashlib.sha256(message.encode()).hexdigest()[:n]


def axial_angle(a,b):
    return abs((a-b+np.pi/2)%np.pi-np.pi/2)


def old_values(stage, direction, row_keys):
    if stage=='ordinary':
        seal=POOL/'SEAL.json'
        assert digest(seal)=='e23d1e43c1ca5115eed70f5e31c99b5a9f99be76728f8eb14ebb5e675d14ed6d'
        path=POOL/(direction+'.npz')
        assert read(seal)['files'][path.name]==digest(path)
        with np.load(path,allow_pickle=False) as z:
            ious=z['candidate_ious']
            return ious[np.arange(len(ious)),z['choices'][:,0,0]], ious.max(1)
    with gzip.open(HIST/f'{stage}_rows.csv.gz','rt') as file:
        hist={r['row']:(float(r['v1_iou']),float(r['pool_best'])) for r in csv.DictReader(file)
              if r['direction']==direction}
    assert set(hist)==set(row_keys)
    return np.array([hist[k] for k in row_keys]).T


def score_direction(job):
    stage,src,tgt,rows,seal=job
    direction=f'{src}_to_{tgt}'
    pin=seal['files'][direction]
    path=PRIVATE/stage/(direction+'.npz')
    assert pin['sha256']==digest(path) and pin['lock']==seal['lock']
    with np.load(path,allow_pickle=False) as data:
        base,yaws,counts=data['base'],data['yaws'],data['counts']
    np.testing.assert_array_equal(base[:,6],yaws[:,0])
    assert len(rows)==pin['n']==len(base)
    keys=[anon(f'{stage}|{direction}|{i}',16) for i in range(len(rows))]
    hist,oracle=old_values(stage,direction,keys)
    assert np.all(hist<=oracle+1e-10)
    iou=np.empty((len(rows),len(METHODS)))
    angular=np.empty_like(iou)
    max_dual=0.
    recordings=[]
    if stage=='ordinary':
        from score183 import truth
    for i,row in enumerate(rows):
        if stage=='ordinary':
            gt=truth(row['metapath'],row['object_index'],row['caption'],tgt,row['suffix'])
        else:
            gt=sensor_truth(tgt,row['metapath'],row['caption'],row['suffix'])
        rec=row.get('recording') or Path(row['metapath']).parent.parent.name
        recordings.append(anon(f'{tgt}|{rec}',12))
        boxes=np.repeat(base[i,None,:],len(METHODS),axis=0)
        boxes[:,6]=np.r_[yaws[i],gt[6]]
        iou[i],err=score_boxes(boxes,gt)
        angular[i]=np.rad2deg(axial_angle(boxes[:,6],gt[6]))
        max_dual=max(max_dual,err)
    np.testing.assert_allclose(iou[:,0],hist,atol=1e-12,rtol=0)
    for tau in (.25,.50):
        np.testing.assert_array_equal(iou[:,0]>tau,hist>tau)
    np.testing.assert_array_equal(iou[counts[:,0]==0,:len(ARMS)],
                                 np.broadcast_to(iou[counts[:,0]==0,:1],(sum(counts[:,0]==0),len(ARMS))))
    filename=RESULTS/f'{stage}_{direction}_rows.csv.gz'
    with gzip.open(filename,'wt',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['row','recording','eligible','native_points','union_points','prior_pool_best',
                         *(f'{m}_iou' for m in METHODS),*(f'{m}_yaw_error_deg' for m in ARMS)])
        for i in range(len(rows)):
            writer.writerow([keys[i],recordings[i],*counts[i],float(oracle[i]),*map(float,iou[i]),
                             *(format(x,'.8f') for x in angular[i,:len(ARMS)])])
    return (src,tgt),dict(iou=iou,angular=angular,recording=np.array(recordings),pool=oracle,
                          eligible=counts[:,0].astype(bool),dual=max_dual,
                          hist=float(np.max(abs(hist-iou[:,0]))),baseline_yaw=pin['native_yaw_max_error'])


def transitions(a,b,t):
    return dict(rescue=int(((a>t)&(b<=t)).sum()),harm=int(((b>t)&(a<=t)).sum()))


def summarize(stage,groups,seal):
    mats={k:v['iou'] for k,v in groups.items()}
    macro=np.mean([measure(x) for x in mats.values()],axis=0)
    allrows={'iou':np.concatenate(list(mats.values()))}
    for k in ('angular','recording','pool','eligible'):
        allrows[k]=np.concatenate([x[k] for x in groups.values()])
    metrics,changes,intervals,coverage,byrec=[],[],[],[],[]
    for (s,t),g in [*groups.items(),(('macro',''),allrows)]:
        d=f'{s}_to_{t}' if t else 'macro'
        mat=g['iou']
        nums=measure(mat) if t else macro
        for i,arm in enumerate(METHODS):
            metrics.append(dict(direction=d,n=len(mat),arm=arm,acc25=nums[i,0],acc50=nums[i,1],miou=nums[i,2]))
        e=g['eligible']
        for a,b in PAIRS:
            i,j=METHODS.index(a),METHODS.index(b)
            x,y=mat[:,i],mat[:,j]
            mask=(x>0)|(y>0)
            q=transitions(x,y,.25)
            r=transitions(x,y,.5)
            changes.append(dict(direction=d,arm=a,baseline=b,n=len(mat),better=int((x>y+1e-10).sum()),
                                worse=int((x<y-1e-10).sum()),rescue25=q['rescue'],harm25=q['harm'],
                                rescue50=r['rescue'],harm50=r['harm'],delta25=nums[i,0]-nums[j,0],
                                delta50=nums[i,1]-nums[j,1],delta_miou=nums[i,2]-nums[j,2]))
            if b=='v1':
                bad_pool=g['pool']<=.5
                coverage.append(dict(direction=d,arm=a,eligible=int(e.sum()),fallback=int((~e).sum()),
                    base_pool_miss50=int(bad_pool.sum()),rescued_outside_pool50=int(((x>.5)&bad_pool).sum()),
                    positive_overlap=int(mask.sum()),base_mae_deg=float(g['angular'][mask,j].mean()),
                    arm_mae_deg=float(g['angular'][mask,i].mean())))
        if t:
            for k in sorted(set(g['recording'])):
                rm=g['recording']==k
                for i,arm in enumerate(ARMS):
                    q=transitions(mat[rm,i],mat[rm,0],.5)
                    byrec.append(dict(direction=d,recording=k,n=int(rm.sum()),arm=arm,
                                      rescue50=q['rescue'],harm50=q['harm']))
    for a,b in PAIRS:
        lo,hi=record_intervals(groups,mats,METHODS.index(a),METHODS.index(b))
        intervals.append(dict(arm=a,baseline=b,lower95_acc50_pp=lo,upper95_acc50_pp=hi))
    for name,records in [('metrics',metrics),('paired',changes),('coverage',coverage),
                         ('intervals',intervals),('recordings',byrec)]:
        csv_file(RESULTS/f'{stage}_{name}.csv',records)
    assert len(allrows['iou'])==SIZES[stage]
    receipt=dict(n=SIZES[stage],prediction_seal=digest(PRIVATE/stage/'SEALED.json'),
        global_seal=digest(PRIVATE/'SEALED_ALL.json'),recording_clusters=len(set(allrows['recording'])),
        historical_baseline_iou_max_error=max(g['hist'] for g in groups.values()),
        independent_iou_max_error=max(g['dual'] for g in groups.values()),
        native_pca_reconstruction_max_error_rad=max(g['baseline_yaw'] for g in groups.values()),
        original_pool='K16' if stage=='ordinary' else 'A256',FINAL_access=False)
    save(RESULTS/f'{stage}_audit.json',receipt)
    return macro.tolist()


def run():
    start=perf_counter()
    all_seals=read(PRIVATE/'SEALED_ALL.json')
    assert set(all_seals)==set(STAGES)
    for stage in STAGES:
        assert all_seals[stage]==digest(PRIVATE/stage/'SEALED.json')
        pin=read(PRIVATE/stage/'SEALED.json')
        assert pin['n']==SIZES[stage] and not pin['task_labels_opened'] and not pin['FINAL_access']
        assert pin['lock']==digest(PRIVATE/stage/'LOCK.json')
        lock=read(PRIVATE/stage/'LOCK.json')
        assert all(digest(HERE/f)==h for f,h in lock['code'].items())
    for stage in STAGES:
        seal=read(PRIVATE/stage/'SEALED.json')
        roster=inputs(stage)
        assert {f'{s}_to_{t}':roster_digest(rs) for (s,t),rs in roster.items()}==read(PRIVATE/stage/'LOCK.json')['rosters']
        jobs=[(stage,s,t,rs,seal) for (s,t),rs in roster.items()]
        with ProcessPoolExecutor(max_workers=6) as workers:
            groups=dict(workers.map(score_direction,jobs))
        metrics=summarize(stage,groups,seal)
        print(stage,dict(zip(METHODS,metrics)),round(perf_counter()-start,1),'s',flush=True)


if __name__=='__main__':
    run()
