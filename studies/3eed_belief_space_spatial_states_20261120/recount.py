"""Independent CSV verification of all ordinary counts, macro accuracy, Brier and selective risk."""
from collections import defaultdict
import csv
import gzip
from statistics import mean
from io_scene import OUT,read,write,digest
from audit_math import verify

def recount(stage):
    cells=defaultdict(list)
    with gzip.open(OUT/(stage+'_all_rows.csv.gz'),'rt') as f:
        for r in csv.DictReader(f):
            p=float(r['prob']);y=int(r['truth'])
            ok=.5 if abs(p-.5)<=1e-14 else float((p>.5)==bool(y))
            assert abs(float(r['hit'])-ok)<1e-12
            cells[r['arm'],r['direction']].append((float(r['confidence']),1-ok,(p-y)**2))
    table=list(csv.DictReader((OUT/(stage+'_metrics.csv')).open()))
    seen=0
    for row in table:
        if row['scope']!='all':continue
        arm,d=row['arm'],row['direction'];values=[]
        for (a,di),v in cells.items():
            if a!=arm or d not in (di,'macro'):continue
            v=sorted(v,reverse=True)
            goal=len(v)/2
            threshold=v[int((len(v)-1)//2)][0]
            sure=[r[1] for r in v if r[0]>threshold]
            tied=[r[1] for r in v if r[0]==threshold]
            risk=(sum(sure)+(goal-len(sure))*mean(tied))/goal
            values.append((1-mean(x[1] for x in v),mean(x[2] for x in v),risk))
        for i,k in enumerate(('acc','brier','risk50')):
            assert abs(mean(v[i] for v in values)-float(row[k]))<2e-10,(arm,d,k)
        seen+=1
    assert seen==49
    report=read(OUT/(stage+'_decision.json'))
    assert sum(len(v) for v in cells.values())==7*report['questions_per_arm']
    return dict(aggregate_rows_recounted=seen,probability_rows_recounted=sum(map(len,cells.values())),
                file=digest(OUT/(stage+'_all_rows.csv.gz')))
if __name__=='__main__':
    stats={'static':recount('static'),'math':verify()}
    if read(OUT/'static_decision.json')['advance']:
        stats['intervention']=recount('intervention')
    write(OUT/'audit.json',stats)
