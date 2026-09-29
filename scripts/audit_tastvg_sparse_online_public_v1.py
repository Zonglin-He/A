"""NumPy-only public aggregation/state chronology readback."""
import json,argparse
from pathlib import Path
import numpy as np

def audit(root):
    read=lambda f:json.loads((root/f).read_text());rows=read('ROWS.json');s=read('SUMMARY.json');progress=read('PROGRESSION.json');checks=0
    assert len(rows)==len({r['parent'] for r in rows})==32;assert [r['position'] for r in rows]==list(range(1,33))
    def check(values,st):
        nonlocal checks
        a=np.array(values,float);assert len(a)==st['n'];assert abs(a.mean()-st['mean'])<1e-12
        rng=np.random.default_rng(20260929);ci=np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]);assert np.allclose(ci,st['ci95'],atol=1e-12,rtol=0);checks+=1
    for i,r in enumerate(rows):
        assert r['expert']==(i%4==0);assert int(np.argmax(r['base_score']))==0
        if i:assert r['arrival_hash']==rows[i-1]['after_hash']
        expert=int(np.argmax(r['teacher_scores']));assert r['selected']['Full Rerank']==expert
        for arm,k in r['selected'].items():assert r['arms'][arm]==r['candidate_metrics'][k]
        if r['expert']:assert r['selected']['Online Slow-Fast']==r['selected']['Budgeted Rerank']==expert
        else:assert r['selected']['Budgeted Rerank']==0 and r['selected']['Online Slow-Fast']==int(np.argmax(r['arrival_scores'])) and r['arrival_hash']==r['after_hash']
    metrics=['sIoU','tIoU','vIoU_corrected']
    for group,st in s.items():
        rr=rows if group=='all' else [r for r in rows if r['expert']==(group=='expert')];assert len(rr)==st['n']
        for a in st['arms']:
            for m in metrics:check([r['arms'][a][m] for r in rr],st['arms'][a][m])
        for c in st['comparisons']:
            a,b=c.split(' - ')
            for m in metrics:
                vv=[r['arms'][a][m]-r['arms'][b][m] for r in rr];check(vv,st['comparisons'][c][m]);assert sum(v<-.05 for v in vv)==st['harms_gt5pp'][c][m]
    for z in progress:
        rr=[r for r in rows[:z['end_position']] if not r['expert']]
        for m in metrics:assert abs(np.mean([r['arms']['Online Slow-Fast'][m]-r['arms']['Budgeted Rerank'][m] for r in rr])-z['cumulative_delta'][m])<1e-12
    return dict(status='pass',cells=32,bootstrap_scalar_checks=checks,expert_positions=8,nonexpert_positions=24,chronology=True,conditional_realized_stream_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(audit(a.directory),indent=2))
