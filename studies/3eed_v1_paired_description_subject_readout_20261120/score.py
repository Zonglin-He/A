"""Open only the registered ordinary labels after the two target directions are jointly sealed."""
import csv
import gzip
import io
import itertools
import sys

from paired import ARMS, HERE, ORDINARY, PUBLIC, WORK, anon, atomic, digest, initialize, np, read
from scope183 import canonical, check_input, ordinary
from target import OUT, BLOCK, execution
from fit import write_csv

sys.path.insert(0,str(HERE.parent/'3eed_v1_component_ablation_20261119'))
from score_ablation import score_boxes


def annotated(row):
    entry = read(row['metapath'])['ground_info'][row['object_index']]
    assert canonical(entry['caption']) == row['caption']
    raw = np.asarray(entry['bbox_3d'], dtype=np.float64)
    assert raw.shape == (9,) and np.isfinite(raw).all() and (raw[3:6]>0).all()
    box = raw[:7].copy()
    if row['platform'] == 'drone':
        box[2] += 1.8
    box = box.astype(np.float32).astype(np.float64)
    if row['platform'] == 'drone':
        box[2] -= 1.8
    return box


def metrics(a):
    return np.array([(a > .25).mean(),(a > .50).mean(),a.mean()])*100


def transitions(a,b):
    """New against reference, strict published IoU cutoffs."""
    result = dict(iou_improved=int((a-b>1e-12).sum()),iou_worse=int((a-b < -1e-12).sum()))
    for t, label in ((.25,'25'),(.50,'50')):
        good, old = a>t, b>t
        result['rescues'+label] = int((good & ~old).sum())
        result['harms'+label] = int((~good & old).sum())
    return result


def compressed_csv(path, rows):
    with path.open('wb') as file:
        with gzip.GzipFile(filename='',mode='wb',fileobj=file,mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)


def process(t, pins, roster):
    hist_root=ORDINARY/'audited_iou_PRIVATE'
    historical_path=hist_root/f'waymo_to_{t}.npz'
    sh=read(hist_root/'SEALED.json')['files'][historical_path.name]['sha256']
    assert digest(historical_path)==sh
    with np.load(historical_path,allow_pickle=False) as hist:
        previous=hist['ious'][:,0]
        previous_oracle=hist['four_best']
    records=[]
    max_error=0.
    max_parity=0.
    for left in range(0,len(roster),BLOCK):
        p=OUT/t/f'{left:05d}.npz'
        pin=pins[p.name]
        assert digest(p)==pin['sha256'] and read(p.with_suffix('.json'))==pin
        with np.load(p,allow_pickle=False) as z:
            ii, picks, scores, boxes=z['ordinals'],z['choices'],z['scores'],z['four']
        assert np.array_equal(ii,np.arange(left,pin['stop'])) and np.array_equal(picks,scores.argmax(-1))
        for i,ord_ in enumerate(ii):
            row=roster[ord_]
            check_input(row)
            exact,error=score_boxes(boxes[i],annotated(row))
            max_error=max(max_error,error)
            values=exact[picks[i]]
            max_parity=max(max_parity,abs(values[0]-previous[ord_]),abs(exact.max()-previous_oracle[ord_]))
            assert abs(values[0]-previous[ord_])<1e-10 and abs(exact.max()-previous_oracle[ord_])<1e-10
            records.append(dict(target=t,rec=row['recording'],sort=anon(t, row['seed'],row['object_index']),
                                values=values,picks=picks[i],oracle=float(exact.max())))
    assert len(records)==len(roster)
    return records,max_error,max_parity


def ci_recording(groups, arm, comp):
    # Exhaustive recording resampling for two and three recordings. Descriptive with so few clusters.
    bytarget={}
    for t, items in groups.items():
        contrib=[]
        for rec in sorted({x['rec'] for x in items}):
            v=np.stack([r['values'] for r in items if r['rec']==rec])
            net=(v[:,arm]>.50).sum()-(v[:,comp]>.50).sum()
            contrib.append((net,len(v)))
        a=np.asarray(contrib)
        choices=list(itertools.product(range(len(a)),repeat=len(a)))
        totals=a[choices,:].sum(1)
        bytarget[t]=100*totals[:,0]/totals[:,1]
    macro=np.mean(np.array(list(itertools.product(*bytarget.values()))),axis=-1)
    return dict(lower=float(np.quantile(macro,.025)),upper=float(np.quantile(macro,.975)),
                exhaustive_resamples=len(macro),covered_recordings={k:len({r['rec'] for r in v})
                                                                    for k,v in groups.items()})


def main():
    cfg=initialize()
    seal=read(OUT/'SEALED.json')
    assert seal['lock']==digest(OUT/'LOCK.json') and read(OUT/'LOCK.json')==execution()
    assert not seal['target_labels_read'] and not seal['FINAL_access'] and seal['n']==5528
    roster=ordinary()
    groups, errors, parities={},[],[]
    for t in cfg['ordinary_counts']:
        rec,e,p=process(t,seal['files'][t],roster[t])
        groups[t]=rec
        errors.append(e)
        parities.append(p)
    all_records=[r for rows in groups.values() for r in rows]
    result, comparisons, individual=[],[],[]
    cohorts={f'waymo_to_{t}':rows for t,rows in groups.items()}
    for t, rows in groups.items():
        recs=sorted({r['rec'] for r in rows},key=lambda v:anon('rec',t,v))
        for i,r in enumerate(recs,1):
            cohorts[f'{t}_recording_{i}']=[x for x in rows if x['rec']==r]
    cohorts['pooled']=all_records
    cohorts['two_equal']=all_records
    for group,rows in cohorts.items():
        values=np.stack([r['values'] for r in rows])
        picks=np.stack([r['picks'] for r in rows])
        vec=[]
        for i,arm in enumerate(ARMS):
            v=(np.mean([metrics(np.stack([x['values'] for x in rr])[:,i]) for rr in groups.values()],0)
                 if group=='two_equal' else metrics(values[:,i]))
            vec.append(v)
            result.append(dict(scope=group,arm=arm,n=len(rows),Acc25=v[0],Acc50=v[1],mIoU=v[2]))
        for new,old in ((1,0),(2,0),(2,1)):
            a,b=values[:,new],values[:,old]
            delta=vec[new]-vec[old]
            comparisons.append(dict(scope=group,arm=ARMS[new],baseline=ARMS[old],n=len(a),
                    Acc25_pp=delta[0],Acc50_pp=delta[1],mIoU_pp=delta[2],
                    switches=int((picks[:,new]!=picks[:,old]).sum()),**transitions(a,b)))
    for i,row in enumerate(sorted(all_records,key=lambda r:r['sort']),1):
        v=row['values']
        individual.append(dict(row=f'T{i:04d}',target=row['target'],
                         v1_iou=float(v[0]),mean_iou=float(v[1]),worst_iou=float(v[2]),
                         four_max_iou=row['oracle'],mean_switch=int(row['picks'][0]!=row['picks'][1]),
                         worst_switch=int(row['picks'][0]!=row['picks'][2]),
                         worst_mean_switch=int(row['picks'][1]!=row['picks'][2])))
    write_csv(PUBLIC/'directions.csv',result)
    write_csv(PUBLIC/'comparisons.csv',comparisons)
    compressed_csv(PUBLIC/'target_all_rows.csv.gz',individual)
    changed=[r for r in individual if r['mean_switch'] or r['worst_switch']]
    write_csv(PUBLIC/'target_switches.csv',changed) if changed else (PUBLIC/'target_switches.csv').write_text(','.join(individual[0])+'\n')
    ci={f'{ARMS[a]}_vs_{ARMS[b]}':ci_recording(groups,a,b) for a,b in ((1,0),(2,0),(2,1))}
    atomic(PUBLIC/'target_audit.json',dict(n=len(all_records),source='waymo',frozen_target=True,
           compute=read(WORK/'HEADS_LOCK.json')['parameters'],predictions_sealed_before_GT=True,
           max_independent_IoU_error=max(errors),max_archived_v1_and_oracle_error=max(parities),
           n_four_boxes_dual_scored=len(all_records)*4,
           max_direct_decoder_replay_error=max(shard['direct_tolerance_max'] for blocks in seal['files'].values() for shard in blocks.values()),
           interval95_Acc50_delta_recording_resample=ci,prediction_sha256=digest(OUT/'SEALED.json'),
           FINAL_access=False,ordinary_seconds=seal['elapsed_seconds']))
    print([r for r in result if r['scope']=='two_equal'],flush=True)


if __name__=='__main__':
    main()
