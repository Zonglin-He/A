"""Paired changes from adding the second direction of the size discrepancy to identical observations."""
from collections import defaultdict
import csv
import gzip
import numpy as np
from paths_shape import OUT, read, save, digest
from score_roles import csv_file,record_intervals
from tables_scene import paired_counts
from verify_shape import load_gz,grouped_local


def run():
    before=load_gz(OUT/'task_rows.csv.gz');after=load_gz(OUT/'symmetric_task_rows.csv.gz')
    old={(v['case'],v['source'],v['arm']):v for v in before}
    asserts=0
    for r in after:
        b=old[r['case'],r['source'],r['arm']]
        for key in ('case','source','target','panel','frame','recording','pool'):
            assert r[key]==b[key]
        if r['arm'] in ('frozen_words','v1'):
            assert all(r[k]==b[k] for k in ('raw_iou','final_iou','iou_s','iou_r','seed_s','seed_r'))
            asserts+=1
        r['before']=float(b['raw_iou']);r['after']=float(r['raw_iou'])
    selected=[r for r in after if r['arm']=='local_shape']
    grouped={k[0]:v for k,v in grouped_local(selected,['panel']).items()}
    grouped['ordinary_combined']=grouped['ordinary_uniform']+grouped['ordinary_relation']
    results=[]
    for panel,items in grouped.items():
        mats={};recs={};per=[]
        for (src,tar),rr in grouped_local(items,['source','target']).items():
            m=np.array([[r['after'],r['before']] for r in rr])
            mats[src,tar]=m;recs[src,tar]=dict(recording=np.array([r['recording'] for r in rr]))
            q=np.array([((m[:,0]>v).mean()-(m[:,1]>v).mean())*100 for v in (.25,.5)])
            per.append(q)
            results.append(dict(panel=panel,direction=src+'_to_'+tar,n=len(m),delta25=q[0],delta50=q[1],
                changed_iou=int((abs(m[:,0]-m[:,1])>1e-10).sum()),**paired_counts(m,0,1),ci_low='',ci_high=''))
        m=np.vstack(list(mats.values()));q=np.mean(per,axis=0);ci=record_intervals(recs,mats,0,1)
        results.append(dict(panel=panel,direction='macro',n=len(m),delta25=q[0],delta50=q[1],
            changed_iou=int((abs(m[:,0]-m[:,1])>1e-10).sum()),**paired_counts(m,0,1),ci_low=ci[0],ci_high=ci[1]))
    csv_file(OUT/'size_rule_comparison.csv',results)
    assert len(old)==len(before)==len(after) and len(selected)==1204 and asserts==2*len(selected)
    save(OUT/'size_rule_verification.json',dict(paired_records=len(selected),compared_all_arms=len(old),
        expected_identical_baseline_rows=asserts,first=digest(OUT/'task_rows.csv.gz'),
        second=digest(OUT/'symmetric_task_rows.csv.gz')))

if __name__=='__main__':run()
