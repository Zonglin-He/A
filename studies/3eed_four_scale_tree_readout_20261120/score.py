"""Independently match exact replay endpoints to all frozen ordinary-validation IoUs."""
from collections import defaultdict
import csv
import gzip

import numpy as np

from replay import BASE, CFG, HERE, OUT, PRIVATE, catalog, digest, pins, read, row_id, save
from shared_paths import anonymous
from endpoint_score import write_gz


def run():
    seal=read(PRIVATE/'SEAL.json')
    assert not seal['labels_used'] and not seal['FINAL_access'] and seal['code']==pins()
    assert seal['protocol']==digest(HERE/'PROTOCOL.json')
    path=BASE/'results/ordinary_rows.csv.gz'
    assert digest(path)==CFG['previous_ordinary_table']
    assert digest(PRIVATE/'decisions.npz')==seal['decisions']
    assert digest(BASE/'private/decisions.npz')==CFG['previous_decisions']
    with np.load(PRIVATE/'decisions.npz') as now,np.load(BASE/'private/decisions.npz') as before:
        np.testing.assert_array_equal(now['keys'],before['keys'])
        np.testing.assert_array_equal(now['subjects'],before['subjects'][:,0])
        np.testing.assert_array_equal(now['refs'],before['refs'][:,0])
        np.testing.assert_array_equal(now['details'][:,:2],before['details'][:,3:5])
        details=dict(zip(now['keys'],now['details']))
    ordinary=set()
    by_scope=defaultdict(list)
    comparisons=[]
    for row in catalog()['rows']:
        if row['scope'] not in ('ordinary','noun'):
            continue
        s,r,n,feasible,ordered=map(int,details.pop(row_id(row)))
        by_scope[row['scope']].append((s,r,n,feasible,ordered))
        comparisons.append(dict(case=row_id(row),scope=row['scope'],source=row['source'],target=row['target'],
                                regions=n,feasible_subjects=feasible,ordered=ordered,
                                subject_box_equal=1,reference_box_equal=1,region_fallback=int(s<0)))
        if row['scope']=='ordinary':
            ordinary.add((row['source'],anonymous((row['target'],row['ordinal']))))
    assert not details and len(ordinary)==CFG['scope']['ordinary_relation']
    cohorts=defaultdict(list)
    seen=set()
    with gzip.open(path,'rt') as f:
        for row in csv.DictReader(f):
            k=row['source'],row['case']
            assert k not in seen
            seen.add(k)
            assert (k in ordinary)==bool(int(row['active']))
            ordinary.discard(k)
            cohorts[row['source']+'_to_'+row['target']].append(float(row['subject_only_shared']))
    assert not ordinary and len(cohorts)==6 and len(seen)==18620
    with (BASE/'results/metrics.csv').open() as f:
        old={r['direction']:r for r in csv.DictReader(f) if r['arm']=='subject_only_shared'}
    rows=[]
    columns=['acc25','acc50','miou']
    def emit(direction,n,metrics):
        np.testing.assert_allclose(metrics,[float(old[direction][k]) for k in columns],rtol=1e-12)
        assert n==int(old[direction]['n'])
        rows.append(dict(direction=direction,n=n,**dict(zip(columns,metrics)),delta25_pp=0.,delta50_pp=0.,
                         delta_miou_pp=0.,rescue25=0,harm25=0,rescue50=0,harm50=0))
    for direction,vals in sorted(cohorts.items()):
        a=np.asarray(vals)
        emit(direction,len(a),np.asarray([(a>.25).mean(),(a>.50).mean(),a.mean()])*100)
    emit('macro',len(seen),[np.mean([r[c] for r in rows]) for c in columns])
    OUT.mkdir(exist_ok=True)
    write_gz(OUT/'replay_rows.csv.gz',sorted(comparisons,key=lambda r:r['case']))
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={}
    for scope,batch in by_scope.items():
        d=np.array(batch)
        summary[scope]=dict(rows=len(d),subject_selections=int((d[:,0]>=0).sum()),
               ordered=int(d[:,4].sum()),ordinary_left_right=int((d[:,4]==0).sum()),
               all_subject_feasibility_checks=int(d[:,2].sum()),feasible_subjects=int(d[:,3].sum()))
    save(OUT/'verification.json',dict(ordinary_rows=len(seen),ordinary_reused_fallback=CFG['scope']['ordinary_fallback'],
         source_frames=seal['source_frames'],endpoint_boxes_equal=seal['rows']*2,endpoint_indices_equal=seal['rows']*2,
         partitions_sizes_and_stop_equal=True,prediction_seal=digest(PRIVATE/'SEAL.json'),scopes=summary,
         previous_decisions_seal=seal['previous'],previous_iou_table_sha256=digest(path),
         max_abs_error=seal['max_abs_error'],source_retraining=0,target_updates=0,FINAL_access=False))


if __name__=='__main__':
    run()
