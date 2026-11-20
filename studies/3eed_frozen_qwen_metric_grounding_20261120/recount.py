"""Recompute dashboard and threshold rescues from the released scalar records only."""
import csv
import json
from pathlib import Path
from collections import Counter
from statistics import mean

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def table(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def check():
    rows=table('all_rows.csv');baseline=table('baseline_rows.csv');metrics=table('metrics.csv');paired=table('paired.csv')
    assert len(rows)==len({r['uid'] for r in rows})==96
    assert Counter(r['target'] for r in rows)==dict.fromkeys(('waymo','drone','quad'),32)
    assert len({(r['uid'],r['source']) for r in baseline})==192
    assert all(r['source']!=r['target'] for r in baseline)
    seen={r['uid']:r for r in rows}
    for r in baseline:
        for a in ('raw','image_ray','depth_only','ray_depth','mismatched_depth'):assert r[a]==seen[r['uid']][a]
    error=0.
    for v in metrics:
        use=baseline if v['arm'] in ('official','registered') else rows
        x=[float(r[v['arm']]) for r in use if v['target']=='macro' or v['target']==r['target']]
        for k,y in [('miou',mean(x)),('acc25',mean(z>.25 for z in x)),('acc50',mean(z>.5 for z in x))]:
            error=max(error,abs(100*y-float(v[k])))
    for v in paired:
        rr=[r for r in rows if v['target']=='macro' or r['target']==v['target']]
        xy=[(float(r[v['arm']]),float(r[v['reference']])) for r in rr]
        for tag,s in [('25',.25),('50',.50)]:
            assert sum(a>s>=b for a,b in xy)==int(v['rescue'+tag])
            assert sum(b>s>=a for a,b in xy)==int(v['harm'+tag])
        assert sum(a>b+1e-10 for a,b in xy)==int(v['improved']) and sum(b>a+1e-10 for a,b in xy)==int(v['worsened'])
    assert error<1e-10
    receipt=dict(unique_frames=len(rows),source_matched_rows=len(baseline),metrics_recomputed=len(metrics),
                 paired_comparisons=len(paired),max_percent_recount_error=error)
    (OUT/'recount.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt)
if __name__=='__main__':check()
