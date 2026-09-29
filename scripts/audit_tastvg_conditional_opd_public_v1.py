"""Public-only independent O3 readout, chronology and parent-cluster statistics."""
import json,argparse
from pathlib import Path
import numpy as np


def audit(root):
    read=lambda n:json.loads((root/n).read_text());rows=read('ROWS.json');summary=read('SUMMARY.json');macro=read('MACRO_ROWS.json');diag=read('DIAGNOSTICS.json');checks=0
    conditions=list(dict.fromkeys(r['condition'] for r in rows));assert len(rows)==80 and len(conditions)==5 and len({r['parent'] for r in rows})==16
    modes=['Conditional Pairwise','Conditional Reverse-KL'];groups={c:[r for r in rows if r['condition']==c] for c in conditions};initial=rows[0]['arrival_hash'][modes[0]]
    order=[r['parent'] for r in groups[conditions[0]]]
    for c,rr in groups.items():
        assert len(rr)==16 and [r['parent'] for r in rr]==order;past=[]
        for i,r in enumerate(rr):
            assert r['stream_position']==i+1 and r['expert']==(i%4==0);assert r['replay_before']==past
            for a in modes:
                if i:assert r['arrival_hash'][a]==rr[i-1]['after_hash'][a]
                else:assert r['arrival_hash'][a]==initial
                if not r['expert']:assert r['selected'][a]==int(np.argmax(r['arrival_scores'][a])) and r['arrival_hash'][a]==r['after_hash'][a]
                else:assert r['diagnostics'][a]['replay_positions']==past
            if r['expert']:
                assert len(set(r['selected'].values()))==1;past=(past+[r['position']])[-4:]
            else:assert r['selected']['Budgeted Rerank']==0 and r['selected']['Linear Slow']==int(np.argmax(r['linear_arrival_scores']))
            assert r['replay_after']==past
            for a,k in r['selected'].items():assert r['arms'][a]==r['candidate_metrics'][k]
        for a in modes:
            nn=[r for r in rr if not r['expert']];assert diag[c][a]['changed']==sum(r['selected'][a]!=0 for r in nn)
    for r in macro:
        rr=[x for x in rows if x['parent']==r['parent']];assert len(rr)==5
        for a in r['arms']:
            for m in r['arms'][a]:assert abs(r['arms'][a][m]-np.mean([x['arms'][a][m] for x in rr]))<1e-12
    groups['macro']=macro
    def check(vals,z):
        nonlocal checks
        a=np.array(vals);assert len(a)==z['n'] and abs(a.mean()-z['mean'])<1e-12;rng=np.random.default_rng(20260929)
        ci=np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]);np.testing.assert_allclose(ci,z['ci95'],rtol=0,atol=1e-12);checks+=1
    for name,ss in summary.items():
        for group,s in ss.items():
            rr=groups[name] if group=='all' else [r for r in groups[name] if r['expert']==(group=='expert')];assert len(rr)==s['n']
            for a in s['arms']:
                for m,z in s['arms'][a].items():check([r['arms'][a][m] for r in rr],z)
            for c in s['comparisons']:
                a,b=c.split(' - ')
                for m,z in s['comparisons'][c].items():
                    vals=[r['arms'][a][m]-r['arms'][b][m] for r in rr];check(vals,z)
                    assert sum(v>1e-12 for v in vals)==s['positive'][c][m] and sum(v<-1e-12 for v in vals)==s['negative'][c][m] and sum(v<-.05 for v in vals)==s['harms_gt5pp'][c][m]
    return dict(status='pass',arrivals=80,unique_parents=16,nonexpert_cells=60,primary_parent_clusters=12,bootstrap_scalar_checks=checks,reset_and_replay_chronology=True,conditional_fixed_trajectories=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(audit(a.directory),indent=2))
