"""Score sealed choices; labels never enter nomination, weights or the decoder."""
import argparse
import csv
import gzip
import hashlib
from pathlib import Path
import sys

import numpy as np

from run_heading import ARMS, BLOCK, CFG, HERE, OUT, ROOT, WORK, digest, inputs, read, roster_hash, save
from assess import measure, paired, record_intervals, score_boxes, sensor_truth
from calibrate import csv_file
from batch_yaw_iou import many_iou


def public_rows(stage):
    if stage=='ordinary':return {}
    history=HERE.parent/'3eed_v1_reserved_seed_verification_20261120'/'results'/f'{stage}_rows.csv.gz'
    with gzip.open(history,'rt') as f:
        return {r['row']:float(r['v1_iou']) for r in csv.DictReader(f)}


def same_box(x,others):
    angle=np.remainder(x[6]-others[:,6]+np.pi/2,np.pi)-np.pi/2
    return (np.max(abs(x[:6]-others[:,:6]),axis=1)<=1e-6)&(np.abs(angle)<=1e-6)


def run(stage):
    work=WORK/stage
    seal=read(work/'SEALED.json')
    lock=read(work/'LOCK.json')
    assert seal['lock']==digest(work/'LOCK.json') and not seal['task_labels_opened'] and not seal['FINAL_access']
    assert all(digest(HERE/name)==h for name,h in lock['code'].items()) and lock['arms']==list(ARMS)
    roster=inputs(stage)
    assert lock['roster']=={f'{s}_to_{t}':roster_hash(rows) for (s,t),rows in roster.items()}
    previous=public_rows(stage)
    checks=dict(v1_history_max_error=0.,dual_iou_max_error=0.,batch_logit_max_error=0.)
    public=[];slices=[];groups={}
    for (s,t),rows in roster.items():
        scope=f'{s}_to_{t}'
        collection={k:[] for k in ('ious','picks','recording','oracle','outside','pool','novel_slate','prior_ious')}
        for off in range(0,len(rows),BLOCK):
            name=f'{scope}/{off:05d}.npz'
            pin=seal['files'][name]
            assert pin['sha256']==digest(work/name) and pin['lock']==seal['lock']
            with np.load(work/name,allow_pickle=False) as stored:
                data=dict(stored)
            for j,pool in enumerate(data['pool']):
                r=rows[off+j];ids=data['ids'][j];picks=data['winners'][j]
                assert np.array_equal(ids[np.arange(len(ARMS)),data['logits'][j].argmax(1)],picks)
                gt=sensor_truth(t,r['metapath'],r['caption'],r['suffix'])
                all_iou=many_iou(pool,gt)
                selected=np.unique(ids)
                check,max_error=score_boxes(pool[selected],gt)
                checks['dual_iou_max_error']=max(checks['dual_iou_max_error'],float(max_error),float(np.max(abs(check-all_iou[selected]))))
                for tau in (.25,.5):assert np.array_equal(check>tau,all_iou[selected]>tau)
                vals=all_iou[picks]
                oracle=all_iou[ids].max(1)
                prior_iou=all_iou[ids[:,0]]
                orig=pool[ids[0]]
                outside=np.array([not same_box(pool[b],orig).any() for b in picks])
                novel_slate=np.array([sum(not same_box(pool[b],orig).any() for b in slate) for slate in ids])
                key=hashlib.sha256(f'{stage}|{scope}|{off+j}'.encode()).hexdigest()[:16]
                rec=r.get('recording') or Path(r['metapath']).parent.parent.name
                rec_key=hashlib.sha256(f'{t}|{rec}'.encode()).hexdigest()[:12]
                if previous:checks['v1_history_max_error']=max(checks['v1_history_max_error'],abs(previous[key]-vals[0]))
                checks['batch_logit_max_error']=max(checks['batch_logit_max_error'],float(data['replay_error'][j]))
                row=dict(row=key,direction=scope,recording=rec_key,pool_best=float(all_iou.max()),
                         pca_yaws=int(data['n_yaw'][j]),pca_all_ms=1000*data['times'][j,0],
                         all_kernels_ms=1000*data['times'][j,1])
                for a,arm in enumerate(ARMS):
                    row.update({arm+'_iou':float(vals[a]),arm+'_switch':int(picks[a]!=picks[0]),
                                arm+'_outside':int(outside[a]),arm+'_best4':float(oracle[a]),
                                arm+'_prior_iou':float(prior_iou[a]),arm+'_new_slots':int(novel_slate[a])})
                public.append(row)
                for k,v in dict(ious=vals,picks=picks,recording=rec_key,oracle=oracle,outside=outside,
                                pool=all_iou.max(),novel_slate=novel_slate,prior_ious=prior_iou).items():
                    collection[k].append(v)
        groups[s,t]={k:np.asarray(v) for k,v in collection.items()}
    assert len(public)==len({r['row'] for r in public})==seal['n']
    assert checks['v1_history_max_error']<1e-9 and checks['dual_iou_max_error']<1e-7
    labels=('Acc25','Acc50','mIoU')
    comparisons=[(i,0) for i in range(1,len(ARMS))]+[(1,i) for i in range(2,len(ARMS))]
    mats={k:v['ious'] for k,v in groups.items()}
    metrics={k:measure(v) for k,v in mats.items()}
    macro=np.mean(list(metrics.values()),axis=0)
    all_data={key:np.concatenate([g[key] for g in groups.values()]) for key in next(iter(groups.values()))}
    arms_out=[];pairs=[];cis=[]
    for key,g in [*groups.items(),('macro',all_data)]:
        scope='macro' if key=='macro' else '_to_'.join(key)
        table=macro if key=='macro' else metrics[key]
        for i,arm in enumerate(ARMS):
            arms_out.append(dict(direction=scope,arm=arm,n=len(g['ious']),**dict(zip(labels,table[i]))))
            changed=g['picks'][:,i]!=g['picks'][:,0]
            any_old_good=g['oracle'][:,0]
            report=dict(direction=scope,arm=arm,n=len(changed),changed=int(changed.sum()),
                        changed_slate=int((g['novel_slate'][:,i]>0).sum()),outside_pick=int(g['outside'][:,i].sum()))
            for tau,token in ((.25,'25'),(.5,'50')):
                old=g['ious'][:,0]>tau;new=g['ious'][:,i]>tau
                a=any_old_good>tau;b=g['oracle'][:,i]>tau;outside=g['outside'][:,i]
                report.update({k+token:int(v.sum()) for k,v in
                    {'hits':new,'first_nominee_hits':g['prior_ious'][:,i]>tau,'pool_covered':g['pool']>tau,
                     'old_four_covered':a,'four_covered':b,'gained_coverage':~a&b,'lost_coverage':a&~b,
                     'outside_opportunity':~a&(g['pool']>tau),'rescued_no_old_good':(~a)&new,
                     'lost_all_old_good':old&(~b),'outside_rescue':outside&(~old)&new,
                     'outside_harm':outside&old&(~new)}.items()})
            slices.append(report)
        for i,j in comparisons:
            pairs.append(dict(direction=scope,arm=ARMS[i],baseline=ARMS[j],
                         **{l+'_pp':float(table[i,k]-table[j,k]) for k,l in enumerate(labels)},
                         **paired(g['ious'],g['picks'],i,j)))
    for i,j in comparisons:
        limits=record_intervals(groups,mats,i,j)
        cis.append(dict(arm=ARMS[i],baseline=ARMS[j],paired_recording_lo50_pp=limits[0],
                        paired_recording_hi50_pp=limits[1],draws=2000))
    prefix=OUT/stage
    for tag,rows in [('metrics',arms_out),('paired',pairs),('coverage',slices),('intervals',cis)]:
        csv_file(prefix.with_name(stage+'_'+tag+'.csv'),rows)
    with gzip.open(OUT/(stage+'_rows.csv.gz'),'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=public[0].keys());w.writeheader();w.writerows(public)
    times=np.array([[r['all_kernels_ms'],r['pca_all_ms']] for r in public])
    save(OUT/(stage+'_audit.json'),dict(n=len(public),arms=list(ARMS),prediction_seal=digest(work/'SEALED.json'),
        code=digest(Path(__file__)),historical_comparisons=len(previous),prediction_seconds=seal['elapsed'],
        median_all_kernels_ms=float(np.median(times[:,0])),median_all_pca_ms=float(np.median(times[:,1])),
        independent_nominee_thresholds_agree=True,**checks,FINAL_access=False))
    if stage=='pilot':
        primary=next(r for r in slices if r['direction']=='macro' and r['arm']==ARMS[1])
        beats={name:bool(macro[1,1]>macro[ARMS.index(name),1]) for name in ('v1','measured_pca','crossed_axes','equal_area_disk')}
        net=primary['outside_rescue50']-primary['outside_harm50']
        save(OUT/'pilot_decision.json',dict(expand=bool(all(beats.values()) and net>0),beats_at_50=beats,
            outside_choices_net50=net,rescue_with_all_original_four_bad50=primary['rescued_no_old_good50'],
            prediction_seal=digest(work/'SEALED.json'),decision_rule=CFG['populations'],FINAL_access=False))
    for a in arms_out:
        if a['direction']=='macro':print(stage,a,flush=True)


if __name__=='__main__':
    cmd=argparse.ArgumentParser();cmd.add_argument('stage',choices=('source','pilot','ordinary'))
    run(cmd.parse_args().stage)
