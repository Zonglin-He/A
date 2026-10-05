"""Post-hoc source/condition and recipient-centered controls; no new inference.

Runnable in the public checkout: python derive_controls.py <results-dir> check.
These controls never alter rank, pair eligibility, basis or the frozen decision.
"""
import sys,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np

def read(p):
    p=Path(p)
    return json.loads(p.read_text()) if p.exists() else json.loads(gzip.decompress(Path(str(p)+'.gz').read_bytes()))

def corr(x,y,w):
    if not w.sum():return None
    x=x-w@x/w.sum();y=y-w@y/w.sum();d=np.sqrt((w@(x*x))*(w@(y*y)))
    return float((w@(x*y))/d) if d>1e-20 else None

def compute(p):
    structure=[];utility=[];coverage=[]
    for b in read(p/'SPECTRUM_INPUTS.json'):
        if b['stream']!='episodic':continue
        for sp in ['search','confirm']:
            ids=[i for i,r in enumerate(b['rows']) if r['split']==sp and r['condition']!='clean'];mm=[b['rows'][i] for i in ids];g=np.array(b['gram'])[np.ix_(ids,ids)];n=len(ids)
            src=[m['source_id'] for m in mm];cond=[m['condition'] for m in mm];ns=len(set(src));nc=len(set(cond));assert n==ns*nc
            grand=np.full((n,n),1/n);ps=np.equal.outer(src,src).astype(float)/nc-grand;pc=np.equal.outer(cond,cond).astype(float)/ns-grand
            total=np.trace(g);mean=np.sum(grand*g);se=np.sum(ps*g);ce=np.sum(pc*g);res=total-mean-se-ce
            structure.append(dict(dataset=b['dataset'],split=sp,cells=n,sources=ns,conditions=nc,total_energy=float(total),mean_energy=float(mean),
                source_energy=float(se),condition_energy=float(ce),interaction_energy=float(res),source_fraction_of_centered=float(se/(total-mean)),condition_fraction_of_centered=float(ce/(total-mean)),interaction_fraction_of_centered=float(res/(total-mean))))
    pairs=read(p/'PAIR_ROWS.json')
    for ds in ['vidstg','hc2']:
        for sp in ['search','confirm']:
            qq=[r for r in pairs if r['dataset']==ds and r['split']==sp and r['condition']!='clean'];q=[r for r in qq if r['cosine'] is not None]
            def srcmean(q,field):
                by=collections.defaultdict(list)
                for r in q:by[r['recipient']].append(r[field])
                return float(np.mean([np.mean(v) for v in by.values()])) if by else None
            coverage.append(dict(dataset=ds,split=sp,all_pairs=len(qq),defined_cosine_pairs=len(q),undefined_cosine_pairs=len(qq)-len(q),
                all_pair_recipient_macro_utility=srcmean(qq,'utility_v'),defined_pair_recipient_macro_utility=srcmean(q,'utility_v'),
                undefined_pair_recipient_macro_utility=srcmean([r for r in qq if r['cosine'] is None],'utility_v')))
            x=np.array([r['cosine'] for r in q]);y=np.array([r['utility_v'] for r in q]);groups=collections.defaultdict(list)
            for i,r in enumerate(q):groups[(r['recipient'],r['condition'])].append(i)
            xc=x.copy();yc=y.copy();groups_with_multiple_donors=0
            for ix in groups.values():
                if len(set(q[i]['donor'] for i in ix))>=2:groups_with_multiple_donors+=1
                xc[ix]-=x[ix].mean();yc[ix]-=y[ix].mean()
            nodes=sorted({r[k] for r in q for k in ['donor','recipient']});idx={s:i for i,s in enumerate(nodes)};di=np.array([idx[r['donor']] for r in q]);ri=np.array([idx[r['recipient']] for r in q]);counts=collections.Counter(ri.tolist());w=np.array([1/counts[int(i)] for i in ri]);point=corr(xc,yc,w)
            boot=[];rng=np.random.default_rng(20261005)
            for _ in range(10000):
                c=rng.multinomial(len(nodes),np.full(len(nodes),1/len(nodes)));v=corr(xc,yc,w*c[di]*c[ri])
                if v is not None:boot.append(v)
            utility.append(dict(dataset=ds,split=sp,recipient_condition_fixed_effect_correlation=point,ci95=np.quantile(boot,[.025,.975]).tolist() if boot else None,
                pairs=len(q),multi_donor_recipient_condition_groups=groups_with_multiple_donors,bootstrap_draws=10000,valid_draws=len(boot),
                bootstrap='common source-node counts; residuals held fixed, no refit'))
    return dict(status='exploratory_posthoc_controls_not_used_for_selection',balanced_episodic_ANOVA=structure,recipient_centered_utility=utility,zero_cosine_coverage=coverage,
        future_leakage_caveat='Online100 matrix source holdout excludes held vectors, not their earlier influence on inherited parameter states. Source-initialized episodic holdout is primary.',
        zero_correction_caveat='Projection fractions undefined for zero vectors; reported source-macro retention is conditional on nonzero corrections. All zeros retained in Gram/coverage.',new_inference=0,new_GT_scoring=0)

def run(p,check=False):
    p=Path(p);result=compute(p);f=p/'POSTHOC_CONTROLS.json'
    if check:
        old=read(f)
        assert old==result
        print('LN_POSTHOC_CONTROL_AUDIT pass',len(result['balanced_episodic_ANOVA']),len(result['recipient_centered_utility']))
    else:
        assert not f.exists();f.write_text(json.dumps(result,indent=2,allow_nan=False));print('LN_POSTHOC_CONTROLS complete')
    return result

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent,check=len(sys.argv)>2 and sys.argv[2]=='check')
