"""Evaluate saved LCIB endpoints after each cycle against the same frozen scope."""
from collections import defaultdict
import sys

import numpy as np

from manifest162 import HERE, WORK, target_rows
from score162 import allowed_gt, put_table, table, verified, weighted
from decode160 import object_indices
from score154 import many_iou


def main(scope):
    ids=verified(scope)
    rows=target_rows()
    out=HERE/'results'/scope
    original={(r['cohort'],int(r['ordinal'])):float(r['lcib']) for r in table(out/'rows.csv')}
    seen=object_indices()
    groups=defaultdict(list)
    for i in ids:
        entry=rows[i]
        c,s,t=(entry[k] for k in ('cohort','source','target'))
        gt=allowed_gt(entry,seen)
        with np.load(WORK/scope/f'{i:05d}.npz',allow_pickle=False) as p:
            scores=many_iou(p['trace_boxes'],gt)
            tr=p['trace']
            assert abs(scores[-1]-original[c,entry['ordinal']])<1e-9
        groups[c,s,t].append((entry['sampling_weight'],scores,tr))
    records=[]
    for (c,s,t),group in sorted(groups.items()):
        w,scores,trace=zip(*group)
        values=np.stack(scores)
        tr=np.stack(trace)
        for j in range(4):
            a,b,d=weighted(values[:,j],w)
            records.append(dict(cohort=c,source=s,target=t,cycle=j,n=len(group),
                                mIoU=a,Acc25=b,Acc50=d,
                                mean_objective=float(np.average(tr[:,j,1],weights=w)),
                                mean_support_L1=float(np.average(tr[:,j,2],weights=w)),
                                mean_center_motion_m=float(np.average(tr[:,j,3],weights=w))))
    for c in sorted({r['cohort'] for r in records}):
        for j in range(4):
            items=[r for r in records if r['cohort']==c and r['cycle']==j]
            assert len(items)==6
            records.append(dict(cohort=c,source='six_equal',target='six_equal',cycle=j,n=sum(r['n'] for r in items),
                **{k:np.mean([r[k] for r in items]) for k in (
                    'mIoU','Acc25','Acc50','mean_objective','mean_support_L1','mean_center_motion_m')}))
    put_table(out/'all_cycle_metrics.csv',records)
    print('Scored trajectory without forward',scope,len(ids),flush=True)


if __name__=='__main__':
    main(sys.argv[1])
