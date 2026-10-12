"""Public anonymous P7 statistics/contrasts audit without model or GT access."""
import collections,gzip,json,math,sys
from pathlib import Path
import numpy as np

def read(p):
    if p.exists():return json.loads(p.read_text())
    with gzip.open(p.with_suffix(p.suffix+'.gz'),'rt') as f:return json.load(f)
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]
def near(a,b):
    if isinstance(a,dict):assert set(a)==set(b);return sum(near(v,b[k]) for k,v in a.items())
    if isinstance(a,list):assert len(a)==len(b);return sum(near(x,y) for x,y in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b
    else:assert abs(a-b)<2e-10*(1+abs(a))
    return 1

def run(root,phase):
    p=Path(root)/'results/stvg_opd_temporal_guided_candidate/2026-10-12'/('P7_'+phase);rr=rows(p/'ROWS.jsonl.gz');s=read(p/'SUMMARY.json');assert len(rr)==512 and len({r['parent_id'] for r in rr})==256
    checks=0
    for ds,data in s.items():
        for subset,blocks in data.items():
            views=([('both',blocks['paired_both_orders'])]+list(blocks['orders'].items())) if phase=='A' else list(blocks.items())
            for order,b in views:
                chosen=[r for r in rr if r['dataset']==ds and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (order=='both' or r['order']==order)];parents=collections.defaultdict(list)
                for r in chosen:parents[r['parent_id']].append(r)
                assert len(parents)==b['parents']=={'all128':128,'previous_P6_32':32,'added96':96}[subset]
                for field,v in b['statistics'].items():
                    vals=[math.fsum(r[field] for r in rs if r[field] is not None)/sum(r[field] is not None for r in rs) for rs in parents.values() if any(r[field] is not None for r in rs)]
                    if not vals:
                        checks+=near(dict(parents=0,mean=None,ci95=None,undefined_parents=len(parents)),v);continue
                    x=np.array(vals);gen=np.random.default_rng(20261012);idx=gen.integers(len(x),size=(10000,len(x)));boot=x[idx].sum(1)/len(x)
                    expected=dict(mean=math.fsum(x)/len(x),ci95=np.quantile(boot,[.025,.975]).tolist(),parent_values=x.tolist(),
                        harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)),positive_parents=int(sum(x>1e-12)),negative_parents=int(sum(x< -1e-12)))
                    if phase=='A':expected.update(min=float(min(x)),max=float(max(x)))
                    else:expected.update(parents=len(x),undefined_parents=len(parents)-len(x))
                    checks+=near(expected,v)
    for r in rr:
        if phase=='A':
            for arm in ['Native','OPD','Direct','Capped_Direct']:
                for m in ['v','t','s','R30','R50']:checks+=near(r[f'{arm}_T1_{m}']-r[f'{arm}_T0_{m}'],r[f'delta_T_{arm}_{m}'])
                assert r[f'{arm}_T0_s']==r[f'{arm}_T1_s']
            for m in ['v','t','s','R30','R50']:checks+=near(r[f'delta_T_OPD_{m}']-r[f'delta_T_Native_{m}'],r[f'interaction_OPD_Native_{m}'])
        else:
            for name,left,right in [('coupling_C_minus_B','C_temporal_guided_OPD','B_posthoc'),('D_minus_B','D_temporal_guided_Direct','B_posthoc'),('C_minus_D','C_temporal_guided_OPD','D_temporal_guided_Direct'),('B_minus_A','B_posthoc','A_original')]:
                for m in ['v','t','s','R30','R50']:checks+=near(r[f'{left}_{m}']-r[f'{right}_{m}'],r[f'{name}_{m}'])
    return dict(status='pass',phase='P7_'+phase,cells=512,parents=256,scalar_checks=checks,scope='anonymous all128/previous32/added96/allorders/complete contrasts and bootstrap',CPU_only=True,GT_read=False)

if __name__=='__main__':print(json.dumps(run(*sys.argv[1:]),allow_nan=False))
