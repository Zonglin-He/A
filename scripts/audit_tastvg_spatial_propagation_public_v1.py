"""Independent scalar-only paired comparison audit, including source bootstrap."""
import sys,json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tastvg_spatial_expansion_public_v1 import run as base_audit

def run(root,prior):
    root,prior=Path(root),Path(prior);r=json.loads((root/'ROWS.json').read_text());old=json.loads((prior/'ROWS.json').read_text());c=json.loads((root/'COMPARISON.json').read_text());out=base_audit(root);checks=0
    byold={(x['parent'],x['condition']):x for x in old}
    for group,comp in c.items():
        by=collections.defaultdict(list)
        for x in r:
            if (x['condition']!='clean' if group=='corruption' else x['condition']=='clean'):by[x['parent']].append(x)
        expected=[]
        for parent,seq in sorted(by.items()):
            paired=[x for x in seq if x['observed_gain_at_s_oracle'] is not None and x['unobserved_gain_at_s_oracle'] is not None]
            val=dict(whole_oracle_gain=np.mean([x['expanded_gain'] for x in seq]),union_gain=np.mean([x['union_oracle_s']-x['native_s'] for x in seq]),oracle_minus_s0=np.mean([x['expanded_gain']-byold[parent,x['condition']]['expanded_gain'] for x in seq]),union_minus_s0=np.mean([x['union_oracle_s']-byold[parent,x['condition']]['union_oracle_s'] for x in seq]),unobserved_paired=np.mean([x['unobserved_gain_at_s_oracle'] for x in paired]) if paired else None,unobserved_minus_s0=np.mean([x['unobserved_gain_at_s_oracle']-byold[parent,x['condition']]['unobserved_gain_at_s_oracle'] for x in paired]) if paired else None)
            saved=next(x for x in comp['source_rows'] if x['parent']==parent)
            for k,v in val.items():
                assert saved[k] is None if v is None else abs(saved[k]-v)<1e-12;checks+=1
            expected.append(val)
        for k,got in comp['metrics'].items():
            a=np.array([x[k] for x in expected if x[k] is not None]);assert len(a)==got['n'];assert abs(a.mean()-got['mean'])<1e-12
            boot=a[np.random.default_rng(20260929).integers(len(a),size=(10000,len(a)))].mean(1);np.testing.assert_allclose(np.quantile(boot,[.025,.975]),got['ci95'],atol=1e-12,rtol=0);checks+=4
    return dict(**out,paired_comparison_checks=checks,comparison_status='pass')

if __name__=='__main__':print(json.dumps(run(sys.argv[1],sys.argv[2]),indent=2))
