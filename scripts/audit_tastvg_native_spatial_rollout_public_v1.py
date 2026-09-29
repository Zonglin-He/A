"""Reconstruct native support oracle and source bootstrap from public scalars."""
import sys,json,collections
from pathlib import Path
import numpy as np

def run(root,prior):
    root,prior=Path(root),Path(prior);read=lambda p:json.loads(p.read_text());rows=read(root/'ROWS.json');summary=read(root/'SUMMARY.json');old={(r['parent'],r['condition']):r for r in read(prior/'ROWS.json')};checks=0
    assert len(rows)==96 and len({r['parent'] for r in rows})==16
    for r in rows:
        v=[m['sIoU'] for m in r['candidate_metrics']];n=v[0];oracle=max(v);six=max(r['six_layer_s']);u=max(oracle,six);p=old[r['parent'],r['condition']]
        expected=dict(native_s=n,rollout_oracle_s=oracle,six_layer_oracle_s=six,union_oracle_s=u,rollout_gain=oracle-n,six_layer_gain=six-n,union_gain=u-n,rollout_minus_six_layer=oracle-six,union_minus_six_layer=u-six,union_minus_s0=u-p['union_oracle_s'],rollout_minus_s0=oracle-p['expanded_oracle_s'])
        for k,val in expected.items():assert abs(r[k]-val)<1e-12;checks+=1
        assert r['best_candidate']==int(np.argmax(v));np.testing.assert_allclose(r['fixed_candidate_gains'],np.array(v)-n,atol=1e-12,rtol=0);checks+=2
    def check_stats(a,got):
        nonlocal checks
        a=np.asarray(a);assert len(a)==got['n'];assert abs(a.mean()-got['mean'])<1e-12
        boot=a[np.random.default_rng(20260929).integers(len(a),size=(10000,len(a)))].mean(1);np.testing.assert_allclose(np.quantile(boot,[.025,.975]),got['ci95'],atol=1e-12,rtol=0);checks+=4
    for group,s in summary.items():
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']==group)];assert len(rr)==s['cells']
        for k,v in s['metrics'].items():
            by=collections.defaultdict(list)
            for r in rr:
                if r[k] is not None:by[r['parent']].append(r[k])
            check_stats([np.mean(by[p]) for p in sorted(by)],v)
        for k,arm in s['fixed_arms'].items():
            a=[np.mean([r['fixed_candidate_gains'][int(k)] for r in rr if r['parent']==p]) for p in sorted({r['parent'] for r in rr})];check_stats(a,arm['gain']);assert sum(x<-.05 for x in a)==arm['sources_worse_gt5pp'];checks+=1
    return dict(status='pass',cells=96,candidates=864,source_count=16,scalar_checks=checks,scope='Saved scalar support and bootstrap; not new model or GT inference')

if __name__=='__main__':print(json.dumps(run(sys.argv[1],sys.argv[2]),indent=2))
