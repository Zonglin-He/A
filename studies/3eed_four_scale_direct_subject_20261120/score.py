"""Score sealed ordinary predictions and reuse exact fallback rows at their original weight."""
import csv
import gzip
from collections import defaultdict

import numpy as np

from predict_direct import ARMS, CFG, OUT, PRIVATE, TABLE, catalog, digest, pins, read, row_id, save
from endpoint_score import write_gz
from scope183 import ordinary
from score_components import recording_interval, summarize
from score_roles import csv_file, score_boxes, sensor_frame_box, sensor_truth
from shared_paths import anonymous


def summarize_scope(name, panels):
    direction_stats = []
    output = []
    def emit(label, n, metrics, counts, ci=None):
        for j, arm in enumerate(ARMS):
            m = metrics[:,j]
            d = m - metrics[:,0]
            output.append(dict(scope=name,direction=label,n=n,arm=arm,acc25=m[0],acc50=m[1],miou=m[2],
                delta25_pp=d[0],delta50_pp=d[1],delta_miou_pp=d[2],
                rescue25=counts[0,j],harm25=counts[1,j],rescue50=counts[2,j],harm50=counts[3,j],
                improved_iou=counts[4,j],worsened_iou=counts[5,j],changed_subject_box=counts[6,j],
                ci50_low=ci[0,j] if ci is not None else '',ci50_high=ci[1,j] if ci is not None else ''))
    for direction,(matrix,changed,recordings) in sorted(panels.items()):
        metrics,counts = summarize(matrix,changed)
        emit('_to_'.join(direction),len(matrix),metrics,counts)
        direction_stats.append((metrics,counts))
    metrics, counts = zip(*direction_stats)
    ci = None
    if name=='all':
        ci = recording_interval({d:(m,r) for d,(m,_,r) in panels.items()})
    emit('macro',sum(len(m) for m,_,_ in panels.values()),np.mean(metrics,axis=0),np.sum(counts,axis=0),ci)
    return output


def run():
    seal = read(PRIVATE/'SEAL.json')
    path = PRIVATE/'decisions.npz'
    assert seal['code']==pins() and seal['decisions']==digest(path) and not seal['labels_used']
    assert digest(TABLE)==CFG['ordinary_iou_table_sha256']
    with np.load(path) as f:
        locked = {k:(s,r,c,o) for k,s,r,c,o in zip(f['keys'],f['subjects'],f['refs'],f['choices'],f['ordered'])}
    table = ordinary()
    effects = {}
    checked_ordered = 0
    scorer_gap = 0.
    for row in catalog()['rows']:
        if row['scope'] not in ('ordinary','noun'):
            continue
        subjects,refs,choices,ordered = locked.pop(row_id(row))
        if row['scope']=='noun':
            assert ordered and np.all(choices==choices[:1])
            for b in (subjects,refs):
                np.testing.assert_array_equal(b, np.repeat(b[:1],len(ARMS),axis=0))
            checked_ordered += 1
            continue
        assert not ordered
        tgt,ix = row['target'],row['ordinal']
        rec = table[tgt][ix]
        pred = np.stack([sensor_frame_box(b,tgt,rec['suffix']) for b in subjects])
        truth = sensor_truth(tgt,rec['metapath'],rec['caption'],rec['suffix'])
        ious, gap = score_boxes(pred,truth)
        scorer_gap = max(scorer_gap,gap)
        changed = (subjects!=subjects[:1]).any(1)
        effects[row['source'],anonymous((tgt,ix))] = (ious,changed,choices)
    assert not locked and len(effects)==1294 and checked_ordered==128
    panels = defaultdict(list)
    rows = []
    seen = set()
    reference_error = 0.
    with gzip.open(TABLE,'rt') as f:
        for rec in csv.DictReader(f):
            baseline = float(rec['subject_only_shared'])
            key = rec['source'],rec['case']
            active = int(rec['active'])
            assert key not in seen and bool(active)==(key in effects)
            seen.add(key)
            ious,changed,choices = effects.pop(key,(np.full(4,baseline),np.zeros(4,bool),np.full((4,2),-1,int)))
            reference_error = max(reference_error,abs(ious[0]-baseline))
            panels[rec['source'],rec['target']].append((ious,changed,active,rec['recording']))
            detail = {k:rec[k] for k in ('source','target','case','recording','active')}
            for j,arm in enumerate(ARMS):
                detail[arm] = ious[j]
                detail[arm+'_box_changed'] = int(changed[j])
            detail['primary_reference_node_changed'] = int(choices[0,1]!=choices[-1,1])
            detail['primary_fallback'] = int(choices[-1,0]<0)
            rows.append(detail)
    assert not effects and len(rows)==18620 and len(panels)==6 and reference_error<1e-10
    metrics = []
    for name in ('all','region','fallback'):
        selected = {}
        for direction,block in panels.items():
            values,changed,active,recordings = map(np.asarray, zip(*block))
            keep = np.ones(len(active),bool) if name=='all' else active==int(name=='region')
            selected[direction] = values[keep],changed[keep],recordings[keep]
        metrics.extend(summarize_scope(name,selected))
    OUT.mkdir(exist_ok=True)
    csv_file(OUT/'metrics.csv',metrics)
    write_gz(OUT/'ordinary_rows.csv.gz',rows)
    save(OUT/'verification.json',dict(ordinary_records=len(seen),relation_replayed=1294,fallback_reused=17326,
         ordered_control=checked_ordered,all_ordered_arms_match_both_baseline_endpoints=True,
         max_dense_subject_mass_difference=seal['max_marginal_error'],max_baseline_iou_difference=reference_error,
         max_two_scorer_difference=scorer_gap,prediction_seal=digest(PRIVATE/'SEAL.json'),ordinary_table_sha256=digest(TABLE),
         ordinary_rows_sha256=digest(OUT/'ordinary_rows.csv.gz'),source_training=0,target_updates=0,FINAL_access=False))
    for r in metrics:
        if r['direction']=='macro' and r['scope']!='fallback':
            print(r['scope'],r['arm'],*[round(r[k],4) for k in ('acc25','acc50','miou','delta50_pp')],r['rescue50'],r['harm50'])


if __name__=='__main__':
    run()
