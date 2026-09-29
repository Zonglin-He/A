"""Independent NumPy readback of all O2 states' public outcomes and cluster stats."""
import json,argparse
from pathlib import Path
import numpy as np


def audit(root):
    read=lambda f:json.loads((root/f).read_text());rows=read('ROWS.json');summary=read('SUMMARY.json');macro=read('MACRO_ROWS.json');progress=read('PROGRESSION.json');diag=read('READOUT_DIAGNOSTIC.json');checks=0
    assert len(rows)==80 and len({r['parent'] for r in rows})==16
    assert [r['position'] for r in rows]==list(range(1,81))
    conditions=list(dict.fromkeys(r['condition'] for r in rows));assert len(conditions)==5
    groups={c:[r for r in rows if r['condition']==c] for c in conditions}
    zero_hash=rows[0]['arrival_hash'];parents=[r['parent'] for r in groups[conditions[0]]]
    for condition,rr in groups.items():
        assert len(rr)==16 and [r['parent'] for r in rr]==parents
        assert rr[0]['arrival_hash']==zero_hash and rr[0]['arrival_norm']==0
        for i,r in enumerate(rr):
            assert r['stream_position']==i+1 and r['expert']==(i%4==0)
            if i:assert r['arrival_hash']==rr[i-1]['after_hash']
            assert int(np.argmax(r['base_score']))==0
            full=int(np.argmax(r['teacher_scores']));assert r['selected']['Full Rerank']==full
            for arm,k in r['selected'].items():assert r['arms'][arm]==r['candidate_metrics'][k]
            if r['expert']:assert r['selected']['Online Slow-Fast']==r['selected']['Budgeted Rerank']==full
            else:assert r['selected']['Budgeted Rerank']==0 and r['selected']['Online Slow-Fast']==int(np.argmax(r['arrival_scores'])) and r['arrival_hash']==r['after_hash']
        nn=[r for r in rr if not r['expert']];d=diag[condition]
        assert d['nonexpert_changed']==sum(r['selected']['Online Slow-Fast']!=0 for r in nn)
        assert d['full_agreement']==sum(r['selected']['Online Slow-Fast']==r['selected']['Full Rerank'] for r in nn)
        assert d['native_full_agreement']==sum(r['selected']['Full Rerank']==0 for r in nn)
        assert d['full_non_native']==sum(r['selected']['Full Rerank']!=0 for r in nn)
        assert d['loss_decreased']==sum(r['diagnostics']['loss_after']<r['diagnostics']['loss_before'] for r in rr if r['expert'])
        for r,z in zip(nn,d['details']):
            base=np.array(r['base_score']);res=np.array(r['arrival_scores'])-base;k=r['selected']['Full Rerank']
            assert abs(z['full_native_gap']-(base[0]-base[k]))<1e-12
            assert abs(z['full_residual_advantage']-(res[k]-res[0]))<1e-12
    for r in macro:
        rr=[x for x in rows if x['parent']==r['parent']];assert len(rr)==5
        for a in r['arms']:
            for m in r['arms'][a]:assert abs(r['arms'][a][m]-np.mean([x['arms'][a][m] for x in rr]))<1e-12
    groups['macro']=macro
    def check(values,st):
        nonlocal checks
        a=np.array(values);assert len(a)==st['n'] and abs(a.mean()-st['mean'])<1e-12
        rng=np.random.default_rng(20260929);ci=np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975])
        np.testing.assert_allclose(ci,st['ci95'],atol=1e-12,rtol=0);checks+=1
    for name,gg in summary.items():
        for group,s in gg.items():
            rr=groups[name] if group=='all' else [r for r in groups[name] if r['expert']==(group=='expert')];assert len(rr)==s['n']
            for a in s['arms']:
                for m in s['arms'][a]:check([r['arms'][a][m] for r in rr],s['arms'][a][m])
            for c in s['comparisons']:
                a,b=c.split(' - ')
                for m in s['comparisons'][c]:
                    vals=[r['arms'][a][m]-r['arms'][b][m] for r in rr];check(vals,s['comparisons'][c][m]);assert sum(v<-.05 for v in vals)==s['harms_gt5pp'][c][m]
    for z in progress:
        rr=[r for r in groups[z['condition']][:z['end_position']] if not r['expert']]
        for m in z['cumulative_delta']:assert abs(z['cumulative_delta'][m]-np.mean([r['arms']['Online Slow-Fast'][m]-r['arms']['Budgeted Rerank'][m] for r in rr]))<1e-12
    return dict(status='pass',arrivals=80,unique_parents=16,streams=5,zero_resets=5,expert_positions=20,nonexpert_positions=60,macro_primary_clusters=12,bootstrap_scalar_checks=checks,conditional_fixed_stream_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(audit(a.directory),indent=2))
