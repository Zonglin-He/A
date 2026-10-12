"""Independent anonymous P8 factorization/partition/parent-statistics audit."""
import collections,gzip,json,math,sys
from pathlib import Path
import numpy as np
def read(p):
    if p.exists():return json.loads(p.read_text())
    with gzip.open(str(p)+'.gz','rt') as f:return json.load(f)
def near(a,b):
    if isinstance(a,dict):assert set(a)==set(b);return sum(near(v,b[k]) for k,v in a.items())
    if isinstance(a,list):assert len(a)==len(b);return sum(near(x,y) for x,y in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b,(a,b)
    else:assert abs(a-b)<2e-10*(1+abs(a)),(a,b)
    return 1
def independent_summary(rr,summary):
    checks=0
    for ds,data in summary.items():
        for subset,orders in data.items():
            for order,block in orders.items():
                chosen=[r for r in rr if r['dataset']==ds and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (order=='both' or r['order']==order)]
                by=collections.defaultdict(list)
                for r in chosen:by[r['parent_id']].append(r)
                assert len(by)==block['parents']=={'all128':128,'previous_P6_32':32,'added96':96}[subset]
                for field,value in block['statistics'].items():
                    vals=[math.fsum(r[field] for r in group if r[field] is not None)/sum(r[field] is not None for r in group) for group in by.values() if any(r[field] is not None for r in group)]
                    if not vals:checks+=near(dict(parents=0,mean=None,ci95=None,undefined_parents=len(by)),value);continue
                    x=np.array(vals);g=np.random.default_rng(20261012);idx=g.integers(len(x),size=(10000,len(x)));boot=x[idx].sum(1)/len(x)
                    expected=dict(parents=len(x),mean=math.fsum(x)/len(x),ci95=np.quantile(boot,[.025,.975]).tolist(),undefined_parents=len(by)-len(x),parent_values=x.tolist(),harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)),positive_parents=int(sum(x>1e-12)),negative_parents=int(sum(x< -1e-12)))
                    checks+=near(expected,value)
    return summary,checks
def run(root):
    p=Path(root)/'results/stvg_opd_support_factorization/2026-10-12'
    with gzip.open(p/'ROWS.jsonl.gz','rt') as f:rr=[json.loads(v) for v in f]
    assert len(rr)==512 and len({r['parent_id'] for r in rr})==256;checks=0
    for r in rr:
        for k in ['v','t','s','R30','R50']:
            for name,a,b in [('temporal_Omega0','B','A'),('temporal_Omega1','C','E'),('observation_I0','E','A'),('observation_I1','C','B')]:checks+=near(r[name+'_'+k],r[a+'_'+k]-r[b+'_'+k])
            checks+=near(r['interaction_'+k],(r['C_'+k]-r['E_'+k])-(r['B_'+k]-r['A_'+k]))
            checks+=near(r['interaction_'+k],(r['C_'+k]-r['B_'+k])-(r['E_'+k]-r['A_'+k]))
        assert r['interaction_s']==r['interaction_t']==0 and r['A_s']==r['B_s'] and r['E_s']==r['C_s']
        checks+=near(r['interaction_v'],sum(r[n+'_interaction_contribution'] for n in ['gained_event','lost_event','common_event']))
    _,n=independent_summary(rr,read(p/'SUMMARY.json'));checks+=n
    return dict(status='pass',scope='all512 anonymous exact2x2 algebra/metric support decomposition/all128-old32-new96/allorders/full paired parent bootstrap',cells=512,parents=256,scalar_checks=checks,CPU_only=True,GT_read=False)
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),allow_nan=False))
