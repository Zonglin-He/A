"""Paired evaluation of the five locked winners with one 3D IoU implementation."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from common160 import (DEV, EXTRA, REC, ORDER, HERE, ROOT, WORK, SPEC, read, rows, sha, canonical, json_new)
from decode160 import ARMS, object_indices, object_index
sys.path.insert(0,str(ROOT/'research/round154'))
from score154 import DATA, ground_truth, many_iou

METRICS=('mIoU','Acc25','Acc50')
COHORTS=(EXTRA,DEV,REC)
COMPARISONS=(('official_span','r158_full'),('native_text_only','r158_full'),
             ('official_span','native_text_only'),('official_span','prepared_axis'),
             ('prepared_axis','prepared_pca'),('prepared_pca','r158_full'))


def csv_write(name,data):
    dest=HERE/'results';dest.mkdir(exist_ok=True)
    with (dest/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)


def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def prior_scalar_rows():
    return {
      DEV:read_csv(ROOT/'research/round158/working_results/all_rows.csv'),
      EXTRA:read_csv(ROOT/'research/round158/transfer_results/all_rows.csv'),
      REC:read_csv(ROOT/'research/round159/results/recording_all_rows.csv')}


def measurable(y):
    return np.stack((y,y>.25,y>.50))


def aggregate(records):
    dir_out=[];overall=[];paired=[];scenes=[];crossings=[]
    by=defaultdict(list)
    for r in records:by[r['cohort'],r['source'],r['target']].append(r)
    for c in COHORTS:
        for s,t in ORDER:
            data=by[c,s,t];n=len(data);weight=np.array([r['sampling_weight'] for r in data])
            assert n and np.isfinite(weight).all() and min(weight)>0
            y={a:np.array([r[a] for r in data]) for a in ARMS}
            meta=dict(cohort=c,source=s,target=t,n=n)
            for a in ARMS:
                x=measurable(y[a])
                vals=np.average(x,weights=weight,axis=1)*100
                dir_out.append(dict(**meta,arm=a,**dict(zip(METRICS,vals))))
                for scene in sorted({r['recording'] for r in data}):
                    keep=np.array([r['recording']==scene for r in data])
                    vals=np.average(x[:,keep],weights=weight[keep],axis=1)*100
                    scenes.append(dict(cohort=c,source=s,target=t,recording=scene,n=int(keep.sum()),arm=a,
                                       **dict(zip(METRICS,vals))))
            for a,b in COMPARISONS:
                delta=measurable(y[b])-measurable(y[a]);gains=np.average(delta,weights=weight,axis=1)*100
                counts={}
                for j,m in enumerate(METRICS):
                    d=delta[j];eps=1e-12 if j==0 else 0
                    counts[m+'_up']=int((d>eps).sum());counts[m+'_down']=int((d<-eps).sum())
                paired.append(dict(**meta,before=a,after=b,**dict(zip(METRICS,gains)),**counts))
            for th in (.25,.5):
                sa=y['official_span']>th;sb=y['r158_full']>th
                crossings.append(dict(**meta,threshold=th,both_correct=int((sa&sb).sum()),
                    official_only=int((sa&~sb).sum()),R158_only=int((~sa&sb).sum()),both_missed=int((~sa&~sb).sum())))
        for a in ARMS:
            group=[r for r in dir_out if r['cohort']==c and r['arm']==a]
            overall.append(dict(cohort=c,arm=a,n=sum(r['n'] for r in group),
                                **{m:np.mean([r[m] for r in group]) for m in METRICS}))
        for a,b in COMPARISONS:
            group=[r for r in paired if r['cohort']==c and r['before']==a and r['after']==b and r['source']!='six_equal']
            paired.append(dict(cohort=c,source='six_equal',target='all',n=sum(r['n'] for r in group),before=a,after=b,
                **{m:np.mean([r[m] for r in group]) for m in METRICS},
                **{k:sum(r[k] for r in group) for k in group[0] if k.endswith(('_up','_down'))}))
    for name,table in [('directions',dir_out),('overall',overall),('paired',paired),('scenes',scenes),
                       ('threshold_crossings',crossings)]:
        csv_write(name+'.csv',table)


def main():
    from eval_lock import verify
    verify()
    fs=read(HERE/'FORWARD_SEAL.json');seal=read(HERE/'PREDICTION_SEAL.json');p=WORK/'BOXES_PRIVATE.npz'
    assert seal['raw_forward_seal']==sha(HERE/'FORWARD_SEAL.json') and sha(p)==seal['sha256'] and fs['protocol']==sha(SPEC)
    assert seal['no_target_boxes_for_prediction'] and not seal['target_class_used_by_R158']
    with np.load(p,allow_pickle=False) as f:
        boxes,winners=f['boxes'],f['winners']
        assert tuple(f['arms'])==ARMS and boxes.shape==(seal['n'],len(ARMS),7)
    registry=rows();idx=object_indices();previous=prior_scalar_rows()
    local=ROOT/'crossplatform/dataset_full/3eed_dataset'
    report=[];scene={};point={};legacy_replayed=0;canonical_changed=0
    for i,e in enumerate(registry):
        r=e['r'];c=e['cohort'];s=e['source'];t=e['target']
        raw=Path(r['metapath']).resolve()
        meta=DATA/raw.relative_to(local)
        if not meta.samefile(raw):raise ValueError('Target scoring index and sensor manifest differ')
        gt=ground_truth(t,str(meta),object_index(e,idx),canonical(r['caption']))
        b=boxes[i].astype(np.float64)
        assert (b[:,3:6]>0).all() and np.isfinite(b).all()
        y=many_iou(b,gt);values=dict(zip(ARMS,map(float,y)))
        if not (np.isfinite(y).all() and y.min()>=0 and y.max()<=1+1e-12):raise ValueError('Invalid outcome')
        prior=previous[c][e['ordinal']-1]
        assert (int(prior['ordinal']),prior['source'],prior['target'])==(e['ordinal'],s,t)
        keys={DEV:('frozen_A_axis','frozen_A','overlap_A_only'),EXTRA:('old_axis','old_top1','overlap'),
              REC:('old_axis','old_pca','r158_pca')}[c]
        changed=canonical(r['caption'])!=r['caption']
        if not changed:
            if max(abs(values[a]-float(prior[k])) for a,k in zip(ARMS[2:],keys))>1e-7:
                raise ValueError('R158 prior values differ')
            legacy_replayed+=1
        else:
            canonical_changed+=1
        sc=scene.setdefault((c,t,r['scene']),1+len(scene))
        group=point.setdefault((c,t,e['group']),1+len(point))
        report.append(dict(cohort=c,ordinal=e['ordinal'],source=s,target=t,recording=sc,point_group=group,
                           sampling_weight=e['sampling_weight'],native_span_and_proxy_differ=int(winners[i,0]!=winners[i,1]),
                           R158_winner_differs_from_official=int(winners[i,0]!=winners[i,-1]),**values))
        if (i+1)%3000==0:print('scored',i+1,flush=True)
    assert len(report)==seal['n'] and legacy_replayed+canonical_changed==len(report)
    csv_write('all_records.csv',report)
    aggregate(report)
    json_new(HERE/'EVALUATION.json',dict(n=len(report),prediction_seal=sha(HERE/'PREDICTION_SEAL.json'),
        common_metric='upright oriented 3D IoU; thresholds strictly greater',cohorts={c:sum(r['cohort']==c for r in report) for c in COHORTS},
        legacy_R158_three_values_reproduced=legacy_replayed,freshly_matched_canonical_descriptions_directed=canonical_changed,
        all_cases_in_denominator=True,FINAL_access=False))
    print('scored all',len(report),flush=True)


if __name__=='__main__':main()
