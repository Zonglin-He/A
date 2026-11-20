"""Score sealed physical boxes against the original inventory and both geometry implementations."""
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import gzip
import hashlib
from pathlib import Path
import sys

import numpy as np

from capture_roles import ARMS, HERE, OUT, PANELS, PRIVATE, ROOT, SOURCES, digest, read, save
from check_math import run as check_math
from assess import measure, normalize, record_intervals, score_boxes, sensor_truth
from batch_yaw_iou import many_iou
from calibrate import csv_file
from crossplatform.candidate_frontend import sensor_frame_box

sys.path[:0] = [str(ROOT/'research/round5/audit'), str(ROOT/'crossplatform/round7'),
                str(ROOT/'crossplatform/round7/experiment'), str(ROOT/'crossplatform/round3/p0/upstream')]
from independent_core_v2 import transform_gt


def anon(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]


@lru_cache(maxsize=512)
def inventory(path, target, view, obj):
    items = read(path)['ground_info']
    main = items[obj]
    labelled = [main['bbox_3d']]
    for item in items:
        labelled.append(item['bbox_3d'])
        labelled.extend(other['bbox_3d_other'] for other in item.get('others', []))
    seen, boxes = set(), []
    for raw in labelled:
        if raw is None or len(raw)<7:
            continue
        key = tuple(map(float, raw[:7]))
        if key in seen or not np.isfinite(key).all() or min(key[3:6])<=0:
            continue
        val = np.array(key if target=='waymo' else (*key, 0., 0.))
        box = transform_gt(val, target, view).astype(float)
        if boxes and many_iou(np.stack(boxes), box).max()>.96:
            continue
        seen.add(key)
        boxes.append(box)
    return np.stack(boxes), normalize(main['caption'])


def owners(shapes, labelled):
    answer = []
    for box in shapes:
        ids = np.flatnonzero(many_iou(labelled, box)>.25) if np.isfinite(box).all() else []
        answer.append(int(ids[0]) if len(ids)==1 else -1)
    return answer


def certificates():
    folder = HERE.parent/'3eed_v1_supervised_instance_grounding_20261119'
    path = folder/'reference_audit_PRIVATE'
    assert digest(path/'BLIND_ROSTER.json')==read(path/'BLIND_SEAL.json')['roster_sha256']
    assert digest(path/'JUDGMENTS.json')==(path/'JUDGMENTS.sha256').read_text().split()[0]
    raw = read(folder/'protocol.json')['evaluation']
    manifest = ROOT/raw['manifest']
    assert digest(manifest)==raw['sha256']
    inputs, labels = read(manifest), read(path/'JUDGMENTS.json')
    cert = {}
    for case in read(path/'BLIND_ROSTER.json'):
        value = labels[str(case['case'])]
        if value is not None:
            inp = inputs[case['ordinal']]
            cert[(str(Path(inp['metapath']).resolve()),normalize(inp['caption']))] = np.array(case['reference_inventory'][str(value)]['box'])
    return cert


def state(owner, desired):
    if desired is None:
        return 'NA'
    if owner<0:
        return 'U'
    return 'T' if owner==desired else 'W'


def score_case(panel, case, entry, source, sealed, cert):
    row = entry['row']
    target, view = row['platform'], int(row['suffix'] or 0)
    tag = f'{panel}_{case:03d}_{source}.npz'
    path = PRIVATE/tag
    assert digest(path)==sealed['files'][tag]
    with np.load(path,allow_pickle=False) as f:
        assert str(f['lock'])==sealed['lock']
        z={k:f[k] for k in f.files if k!='lock'}
    obj=int(row['object_index'])
    gt, caption = inventory(row['metapath'],target,view,obj)
    assert caption==normalize(row['caption'])
    gt_sensor=sensor_truth(target,str(Path(row['metapath']).resolve()),row['caption'],row['suffix'])
    check_gt=many_iou(sensor_frame_box(gt[0],target,view)[None],gt_sensor)[0]
    assert abs(check_gt-1.)<1e-7
    before=int(z['choices'][0])
    np.testing.assert_array_equal(z['raw'],z['slate'][z['choices']])
    np.testing.assert_array_equal(z['boxes'][ARMS.index('ordered_raw')],z['raw'][-1])
    pool_best=float(many_iou(z['pool'],gt[0]).max())
    all_best=max(pool_best,float(many_iou(z['other_pools'],gt[0]).max(initial=0.)))
    native_axis = z['pool'].copy()
    all_axis = z['other_pools'].copy()
    native_axis[:,6] = 0.
    all_axis[:,6] = 0.
    native_union = max(pool_best,float(many_iou(native_axis,gt[0]).max()))
    strict_union = max(all_best,native_union,float(many_iou(all_axis,gt[0]).max(initial=0.)))
    role=None
    if panel=='source_proxy':
        role=int(entry['ref_id'])
        assert 0<role<len(gt)
    else:
        key=(str(Path(row['metapath']).resolve()),caption)
        if key in cert and len(entry['parsed']['graph']['entities'])==2:
            gt_sensor_boxes=np.stack([sensor_frame_box(b,target,view) for b in gt])
            ids=np.flatnonzero(many_iou(gt_sensor_boxes,cert[key])>.96)
            if len(ids)==1 and ids[0]>0:
                role=int(ids[0])
    predicted=np.vstack((z['boxes'],z['slate']))
    native,err1=score_boxes(predicted,gt[0])
    sensor=np.stack([sensor_frame_box(b,target,view) for b in predicted])
    cross,err2=score_boxes(sensor,gt_sensor)
    np.testing.assert_allclose(native,cross,atol=1e-8,rtol=0.)
    for threshold in (.25,.5):
        np.testing.assert_array_equal(native>threshold,cross>threshold)
    ious,unmodified=np.split(native,[len(ARMS)])
    ref_ids=owners(z['refs'],gt)
    orig_ids=owners(z['slate'],gt)
    refined_ids=owners(z['boxes'],gt)
    best_slot=float(unmodified.max())
    assert best_slot<=pool_best+1e-11
    metadata=dict(panel=panel,group=('own_proxy' if source==target else 'cross_proxy') if panel=='source_proxy' else panel,
                  source=source,target=target,case=anon(f'{panel}|{case}'),
                  recording=anon(str(Path(row['metapath']).resolve().parent.parent)),
                  active=int(z['active']),ref_label_kind=('source_proxy' if panel=='source_proxy' else 'independent') if role is not None else 'unresolved',
                  best_slate=best_slot,best_a256=pool_best,best_all=all_best,
                  best_a256_axis_and_pca=native_union,best_all_axis_and_pca=strict_union,
                  baseline_owner=state(orig_ids[before],0),baseline_iou=float(ious[0]))
    records=[]
    for i, arm in enumerate(ARMS):
        slot=int(z['choices'][i])
        records.append(dict(**metadata,arm=arm,iou=float(ious[i]),raw_iou=float(unmodified[slot]),
                            slot_switched=int(slot!=before),same_raw_instance=int(orig_ids[slot]==orig_ids[before] and orig_ids[slot]>=0),
                            raw_subject=state(orig_ids[slot],0),subject=state(refined_ids[i],0),
                            reference=state(ref_ids[i],role),shape_weight=float(z['fraction'][i])))
    return records,max(err1,err2,float(np.max(abs(cross-native))))


def counts(a,b,t):
    return int(((a>t)&(b<=t)).sum()),int(((a<=t)&(b>t)).sum())


def summarize(stream):
    metrics, paired, geometry, identity, intervals=[],[],[],[],[]
    for group in ('own_proxy','cross_proxy','natural170','natural171'):
        cohort=[r for r in stream if r['group']==group]
        buckets=defaultdict(list)
        for row in cohort:
            buckets[row['source'],row['target']].append(row)
        derived={}
        for key,rows in buckets.items():
            by={arm:[r for r in rows if r['arm']==arm] for arm in ARMS}
            assert all([r['case'] for r in g]==[r['case'] for r in by['v1']] for g in by.values())
            mat=np.column_stack([[r['iou'] for r in by[arm]] for arm in ARMS])
            derived[key]=(by,mat)
        macro=np.mean([measure(mat) for _,mat in derived.values()],axis=0)
        joined={arm:[r for by,_ in derived.values() for r in by[arm]] for arm in ARMS}
        jobs=[(s+'_to_'+t,by,mat,measure(mat)) for (s,t),(by,mat) in derived.items()]
        jobs.append(('macro',joined,np.concatenate([m for _,m in derived.values()]),macro))
        for direction,by,mat,tbl in jobs:
            for i,arm in enumerate(ARMS):
                rows=by[arm]
                infos=dict(group=group,direction=direction,arm=arm,n=len(rows))
                values=mat[:,i]
                metrics.append(dict(**infos,acc25=tbl[i,0],acc50=tbl[i,1],miou=tbl[i,2],
                                    switched=sum(r['slot_switched'] for r in rows),active=sum(r['active'] for r in rows)))
                raw=np.array([r['raw_iou'] for r in rows])
                ok=np.array([r['baseline_owner']==r['raw_subject']=='T' for r in rows])
                all_a=np.array([r['best_a256'] for r in rows])
                all_pool=np.array([r['best_all'] for r in rows])
                hard_pool=np.array([r['best_all_axis_and_pca'] for r in rows])
                rescue,harm=counts(values[ok],raw[ok],.50)
                error=np.array([r['subject']!='T' for r in rows])
                geometry.append(dict(**infos,same_verified_subject=int(ok.sum()),rescue50=rescue,harm50=harm,
                   lost_resolved_subject=int((ok&error).sum()),shape_moved=sum(r['shape_weight']>1e-10 for r in rows),
                   verified_misses_all_a256=int((ok&(all_a<=.50)).sum()),
                   verified_rescue_outside_a256=int((ok&(values>.50)&(all_a<=.50)).sum()),
                   verified_rescue_outside_all=int((ok&(values>.50)&(all_pool<=.50)).sum()),
                   verified_rescue_outside_all_axis_and_pca=int((ok&(values>.50)&(hard_pool<=.50)).sum()),
                   all_rescue_outside_a256=int(((values>.50)&(all_a<=.50)).sum()),
                   all_rescue_outside_all=int(((values>.50)&(all_pool<=.50)).sum())))
                transitions=Counter(r['baseline_owner']+r['raw_subject'] for r in rows)
                known=[r for r in rows if r['reference']!='NA']
                identity.append(dict(**infos,**{a+'_to_'+b:transitions[a+b] for a in ('T','W','U') for b in ('T','W','U')},
                    distinct_resolved_switches=sum(r['slot_switched'] and not r['same_raw_instance'] and r['baseline_owner']!='U' and r['raw_subject']!='U' for r in rows),
                    ref_adjudicable=len(known),ref_T=sum(r['reference']=='T' for r in known),
                    both_T=sum(r['reference']==r['raw_subject']=='T' for r in known)))
                for control in dict.fromkeys(('v1',*([a for a in ARMS if a!=arm] if arm=='ordered_full' else []))):
                    j=ARMS.index(control)
                    a,b=counts(values,mat[:,j],.25)
                    c,d=counts(values,mat[:,j],.5)
                    paired.append(dict(**infos,control=control,delta_acc25=tbl[i,0]-tbl[j,0],delta_acc50=tbl[i,1]-tbl[j,1],
                                       delta_miou=tbl[i,2]-tbl[j,2],rescue25=a,harm25=b,rescue50=c,harm50=d))
        group_keys={k:{'recording':np.array([r['recording'] for r in by['v1']])} for k,(by,_) in derived.items()}
        mats={k:mat for k,(_,mat) in derived.items()}
        for j,arm in enumerate(ARMS[:-1]):
            lo,hi=record_intervals(group_keys,mats,len(ARMS)-1,j)
            intervals.append(dict(group=group,control=arm,lo_acc50_pp=lo,hi_acc50_pp=hi))
    for name,values in [('metrics',metrics),('paired',paired),('identity',identity),('geometry',geometry),('intervals',intervals)]:
        csv_file(OUT/(name+'.csv'),values)


def run():
    sealed=read(PRIVATE/'SEAL.json')
    lock=read(PRIVATE/'LOCK.json')
    assert not sealed['original_role_labels_used_for_prediction'] and sealed['n']==len(sealed['files'])==501
    assert not sealed['FINAL_access'] and sealed['lock']==digest(PRIVATE/'LOCK.json')
    assert sealed['target_updates']==0 and lock['protocol']==digest(HERE/'PROTOCOL.json')
    assert all(digest(HERE/f)==v for f,v in lock['code'].items())
    cert=certificates()
    stream=[]
    seen=set()
    err=0.
    for panel,path in PANELS.items():
        for case,entry in enumerate(read(path)):
            for source in SOURCES:
                if panel!='source_proxy' and entry['target']==source:
                    continue
                records,difference=score_case(panel,case,entry,source,sealed,cert)
                stream.extend(records)
                err=max(err,difference)
                seen.add(f'{panel}_{case:03d}_{source}.npz')
    assert seen==set(sealed['files']) and len(stream)==501*len(ARMS)
    with gzip.open(OUT/'all_rows.csv.gz','wt',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=stream[0].keys())
        writer.writeheader()
        writer.writerows(stream)
    summarize(stream)
    save(OUT/'audit.json',dict(n=501,arm_rows=len(stream),collection_seconds=sealed['elapsed'],max_independent_iou_error=err,
         prediction_seal=digest(PRIVATE/'SEAL.json'),math=check_math(),source_assets=lock['source_assets'],FINAL_access=False))


if __name__=='__main__':
    run()
