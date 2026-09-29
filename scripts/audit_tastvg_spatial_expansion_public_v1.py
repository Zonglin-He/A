"""Reconstruct S0 oracle statistics from public scalar-only rows."""
import sys,json,collections
from pathlib import Path
import numpy as np

def run(root):
    root=Path(root);rows=json.loads((root/'ROWS.json').read_text());summaries=json.loads((root/'SUMMARY.json').read_text());checks=0
    assert len(rows)==96 and len({r['parent'] for r in rows})==16
    for r in rows:
        vals=[x['sIoU'] for x in r['step_metrics']];native=vals[0];exp=max(vals);six=max(r['six_layer_s'])
        expected=dict(native_s=native,expanded_oracle_s=exp,six_layer_oracle_s=six,union_oracle_s=max(exp,six),expanded_gain=exp-native,six_layer_gain=six-native,expanded_minus_six_layer=exp-six,union_minus_six_layer=max(exp,six)-six,**{f'step{j}_gain':vals[j]-native for j in range(1,4)})
        for k,v in expected.items():assert abs(r[k]-v)<1e-12;(checks:=checks+1)
        assert r['best_step']==int(np.argmax(vals)) and r['expanded_gain']>=0 and r['six_layer_gain']>=0
    for group,s in summaries.items():
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
        assert s['cells']==len(rr)
        for key,got in s['metrics'].items():
            by=collections.defaultdict(list)
            for r in rr:
                if r[key] is not None:by[r['parent']].append(r[key])
            # Public Q01 order equals original integer order.
            a=np.array([np.mean(by[p]) for p in sorted(by)])
            if not len(a):assert got['n']==0;continue
            boot=a[np.random.default_rng(20260929).integers(len(a),size=(10000,len(a)))].mean(1)
            expect={'mean':a.mean(),'median':np.median(a),'min':a.min(),'max':a.max()}
            for k,v in expect.items():assert abs(got[k]-v)<1e-12;checks+=1
            assert np.allclose(got['ci95'],np.quantile(boot,[.025,.975]),atol=1e-12,rtol=0);checks+=2
    return dict(status='pass',cells=96,source_count=16,scalar_checks=checks,scope='Scalar reconstruction, not RVOS/TA model rerun')

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
