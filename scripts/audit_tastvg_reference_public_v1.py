"""Recompute all public pair decisions, task utilities and source bootstrap."""
import sys,json,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def run(folder):
    checks=0
    def close(a,b):
        nonlocal checks
        assert abs(a-b)<1e-12,(a,b);checks+=1
    for ds in ['vidstg','hc2']:
        rows=json.loads((folder/ds/'ROWS.json').read_text());summary=json.loads((folder/ds/'SUMMARY.json').read_text())
        assert len(rows)==30 and len({r['source_id'] for r in rows})==10
        for r in rows:
            for strategy in ['Uniform','Routed']:
                rewards=r[strategy+'_rewards'];rr=[0.]*9 if rewards is None else rewards
                selected=0 if rewards is None else max(range(9),key=lambda i:rr[i]);assert selected==r[strategy+'_selected'];checks+=1
                for scope in ['fixed','joint']:
                    u=r[scope+'_utilities'];close(r[strategy+'_'+scope+'_v'],u[selected]);close(r[strategy+'_'+scope+'_gain'],u[selected]-u[0]);close(r[strategy+'_'+scope+'_regret'],max(u)-u[selected])
                    strict=decisive=0;total=0.
                    for p in r[strategy+'_'+scope+'_pairs']:
                        i,j=p['i'],p['j'];d,g=rr[i]-rr[j],u[i]-u[j];close(d,p['reward_difference']);close(g,p['utility_difference'])
                        a,b=int(d>1e-12)-int(d< -1e-12),int(g>1e-12)-int(g< -1e-12)
                        assert a==p['expert_sign'] and b==p['GT_sign'];checks+=2
                        if b:
                            score=.5 if a==0 else float(a==b);close(score,p['accuracy']);strict+=1;decisive+=a!=0;total+=score
                        else:assert p['accuracy'] is None
                    assert strict==r[strategy+'_'+scope+'_strict_pairs'];checks+=1
                    if strict:close(total/strict,r[strategy+'_'+scope+'_pairwise']);close(decisive/strict,r[strategy+'_'+scope+'_decisive'])
                    else:assert r[strategy+'_'+scope+'_pairwise'] is None
            for k,v in r.items():
                if k.startswith('delta_'):
                    f=k[6:];a,b=r['Uniform_'+f],r['Routed_'+f]
                    if a is None or b is None:assert v is None
                    else:close(v,b-a)
        for group,z in summary.items():
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')]
            assert len(rr)==z['cells']
            for field,m in z['metrics'].items():
                take=[r for r in rr if r[field] is not None];sources=collections.defaultdict(list)
                for r in take:sources[r['source_id']].append(r[field])
                vector=np.asarray([np.mean(sources[s]) for s in sorted(sources)]);close(vector.mean(),m['mean']);close(np.mean([r[field] for r in take]),m['query_macro'])
                rng=np.random.default_rng(20261001);boot=np.concatenate([vector[rng.integers(0,len(vector),(100,len(vector)))].mean(1) for _ in range(100)])
                ci=np.quantile(boot,[.025,.975]);close(ci[0],m['ci95'][0]);close(ci[1],m['ci95'][1])
            dv=np.asarray([r['delta_fixed_v'] for r in rr]);close(np.maximum(dv,0).mean()*100,z['cell_gross_gain_pp']);close(-np.minimum(dv,0).mean()*100,z['cell_gross_loss_pp'])
    result=dict(status='pass',checks=checks,raw_private_assets_required=False);print(json.dumps(result));return result
if __name__=='__main__':run(Path(sys.argv[1]))
