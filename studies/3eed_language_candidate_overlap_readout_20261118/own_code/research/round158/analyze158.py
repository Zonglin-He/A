"""Independent reconstruction of published metrics and post-seal gain/loss groups."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from math import fsum

HERE=Path(__file__).resolve().parent
def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    d=HERE/'working_results';x=HERE/'transfer_results'
    a=read(d/'all_rows.csv'); b=read(x/'all_rows.csv')
    directions=sorted(set((r['source'],r['target']) for r in a))
    compare=read(d/'macro.csv');extra=read(x/'macro.csv');maxerr=0.; negatives={}
    if len(directions)!=6 or len(a)!=1798 or len(b)!=5942:raise ValueError('cohort size')
    for rows,arms,expected,study in [(a,('joint_zero','overlap_reader','overlap_wrong_B','overlap_A_only','role_risk','context_overlap','context_competition_only'),compare,'main'),
            (b,('old_top1','overlap','old_axis','overlap_axis'),extra,'extra')]:
        for arm in arms:
            by_dir=[]
            for src,tgt in directions:
                group=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
                w=[float(r.get('sampling_weight',1)) for r in group]
                y=[float(r[arm]) for r in group]
                by_dir.append([100*fsum(z*v for z,v in zip(w,measure))/fsum(w)
                               for measure in (y,[v>.25 for v in y],[v>.5 for v in y])])
            s=[r for r in expected if r['arm']==arm and (r.get('weighting')=='ordinary' if study=='extra' else
                 r['group']=='six_equal' and r['aggregation']=='design_weighted')]
            if len(s)!=1: raise ValueError('aggregation mismatch')
            maxerr=max(maxerr,*(abs(mean(q[j] for q in by_dir)-float(s[0][k])) for j,k in enumerate(('mIoU','Acc25','Acc50'))))
    if maxerr>1e-10:raise ValueError('independent scalar aggregation')
    sizes={dr:fsum(float(r['sampling_weight']) for r in a if (r['source'],r['target'])==dr) for dr in directions}
    pool=defaultdict(list);tails=[]
    for r in a:
        if r['old_scores_risk']!=r['overlap_reader']:raise ValueError('live changed outcome')
        if r['has_B']=='False' and len({r[k] for k in ('overlap_reader','overlap_A_only','overlap_wrong_B')})!=1:
            raise ValueError('no B invariance')
        old,new=map(float,(r['joint_zero'],r['overlap_reader']))
        iou=float(r['current_old_new_overlap'])
        label='same_query' if r['new_top1']=='False' else 'disjoint' if iou<=1e-12 else 'at_least_half_overlap' if iou>=.5 else 'partial_overlap'
        contribution=100*float(r['sampling_weight'])/6/sizes[(r['source'],r['target'])]*(new-old)
        pool[label].append((new-old,contribution))
        tails.append(dict(ordinal=int(r['ordinal']),source=r['source'],target=r['target'],has_B=r['has_B']=='True',
                          old_iou=old,new_iou=new,box_overlap=iou,category=label,contribution_pp=contribution))
    by_geometry={}
    for k,vals in pool.items():
        dx=[x for x,_ in vals];contrib=[c for _,c in vals]
        by_geometry[k]=dict(n=len(vals),better=sum(v>1e-12 for v in dx),worse=sum(v< -1e-12 for v in dx),
                            gain_pp=fsum(max(0,c) for c in contrib),loss_pp=fsum(min(0,c) for c in contrib),net_pp=fsum(contrib))
    pair=next(r for r in read(d/'paired.csv') if r['scope']=='all' and r['control']=='joint_zero')
    if abs(fsum(v['net_pp'] for v in by_geometry.values())-float(pair['net_pp']))>1e-7:
        raise ValueError('contribution denominator')
    tails.sort(key=lambda x:x['contribution_pp'])
    payload=dict(status='PASS',independent_scalar_max_error=maxerr,all_negative_rows_preserved=True,
                 by_geometry=by_geometry,largest_positive=tails[-3:][::-1],largest_negative=tails[:3],
                 caveat='Axis-aligned inter-box overlap describes geometry, not independently checked instance identity.')
    (d/'SCALAR_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(by_geometry))

if __name__=='__main__':main()
