"""Independent anonymous scalar reconstruction, NumPy only."""
import argparse,json,collections
from pathlib import Path
import numpy as np

def audit(root):
    rows=json.loads((root/'ROWS.json').read_text());summary=json.loads((root/'SUMMARY.json').read_text());excess=json.loads((root/'EXCESS.json').read_text());checks=0
    assert len(rows)==256 and len({r['parent'] for r in rows})==16
    def check(rr,fn,z):
        nonlocal checks
        d=collections.defaultdict(list)
        for r in rr:d[r['parent']].append(fn(r))
        a=np.array([sum(d[k])/len(d[k]) for k in sorted(d)])
        assert len(a)==z['n'];assert abs(float(a.mean())-z['mean'])<1e-12
        rng=np.random.default_rng(20260929);ci=np.quantile(a[rng.integers(0,len(a),(10000,len(a)))].mean(1),[.025,.975])
        assert np.allclose(ci,z['ci95'],atol=1e-12,rtol=0);checks+=1
    metrics=['sIoU','tIoU','vIoU_corrected']
    for r in rows:
        assert r['selected']==int(np.argmax(r['candidate_scores']))
        assert r['oracle']==int(np.argmax([v['tIoU'] for v in r['candidate_metrics']]))
        assert r['arms']['Frozen']==r['candidate_metrics'][0] and r['arms']['Rerank']==r['candidate_metrics'][r['selected']]
    for name,z in summary.items():
        rr=[r for r in rows if (r['condition']!='clean' if name=='transient' else r['condition']==name)]
        assert len(rr)==z['cells']
        for a in z['arms']:
            for m in metrics:check(rr,lambda r:r['arms'][a][m],z['arms'][a][m])
        for c in z['comparisons']:
            a,b=c.split(' - ')
            for m in metrics:
                check(rr,lambda r:r['arms'][a][m]-r['arms'][b][m],z['comparisons'][c][m])
                assert sum(r['arms'][a][m]-r['arms'][b][m]<-.05 for r in rr)==z['harms_gt5pp'][c][m]
        for a in ['Hard','OPD']:assert sum(r['diagnostics'][a]['loss_decreased'] for r in rr)==z['loss_decreased'][a]
    clean={r['parent']:r for r in rows if r['condition']=='clean'};rr=[r for r in rows if r['condition']!='clean']
    for a in excess:
        for m in metrics:check(rr,lambda r:(r['arms'][a][m]-r['arms']['Frozen'][m])-(clean[r['parent']]['arms'][a][m]-clean[r['parent']]['arms']['Frozen'][m]),excess[a][m])
    return dict(status='pass',cells=256,parents=16,bootstrap_scalar_checks=checks,selection_and_harm_counts=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(audit(a.directory),indent=2))
