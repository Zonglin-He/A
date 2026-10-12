"""Independent source-state arithmetic and anonymous paired P9-P0 statistics."""
import collections,gzip,json,math,sys,warnings
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr,rankdata
BOOT=10000;SEED=20261013;EPS=1e-12
def near(a,b):
    if isinstance(a,dict):assert set(a)==set(b),(set(a),set(b));return sum(near(v,b[k]) for k,v in a.items())
    if isinstance(a,list):assert len(a)==len(b);return sum(near(v,w) for v,w in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b,(a,b)
    else:assert abs(a-b)<2e-10*(1+abs(a)),(a,b)
    return 1
def independent_states(z):
    arrays=[]
    for state in z['initial_states']+[z['source_initial_state']]:
        ln=[]
        for k in sorted(state):
            v=state[k].numpy().reshape(-1);assert len(v)==256 and np.isfinite(v).all()
            if 'query_residual' in k:assert not np.count_nonzero(v)
            else:ln.extend(map(float,v))
        assert len(ln)==1536;arrays.append(ln)
    x,y,s=arrays;d=math.sqrt(math.fsum((a-b)**2 for a,b in zip(x,y)));source_norm=math.sqrt(math.fsum(a*a for a in s))
    expected=dict(LN_difference_L2=d,LN_difference_RMS=math.sqrt(math.fsum((a-b)**2 for a,b in zip(x,y))/1536),LN_difference_relative_source_norm=d/source_norm,LN_order1_source_L2=math.sqrt(math.fsum((a-b)**2 for a,b in zip(x,s))),LN_order2_source_L2=math.sqrt(math.fsum((a-b)**2 for a,b in zip(y,s))),query_residuals_both_exact_zero=True,initial_states_identical=x==y,before_box_displacement=z['stats']['before_box_displacement'])
    return near(expected,z['stats'])+3*1792
def interval(v):
    a=np.asarray(v,float);finite=a[np.isfinite(a)];return dict(ci95=np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None,valid_resamples=len(finite),undefined_resamples=len(a)-len(finite))
def effects(rr,o):
    a=[r['order1_'+o] for r in rr];b=[r['order2_'+o] for r in rr]
    def label(v):return 'unknown' if v is None else 'positive' if v>EPS else 'negative' if v< -EPS else 'neutral'
    matrix=collections.Counter(label(x)+'_'+label(y) for x,y in zip(a,b));known=[(x,y) for x,y in zip(a,b) if x is not None and y is not None];nn=[(x,y) for x,y in known if abs(x)>EPS and abs(y)>EPS];flips=[(x,y) for x,y in nn if x*y<0];n=len(known)
    out=dict(total_parents=len(rr),defined_pairs=n,unknown_pairs=len(rr)-n,both_nonneutral_pairs=len(nn),transition_matrix=dict(matrix),sign_flip_parents=len(flips),sign_flip_all_parent_rate=len(flips)/len(rr),sign_flip_nonneutral_rate=len(flips)/len(nn) if nn else None,flip_both_over1pp=sum(abs(x)>=.01 and abs(y)>=.01 for x,y in flips),flip_both_over5pp=sum(abs(x)>=.05 and abs(y)>=.05 for x,y in flips),flip_effect_difference_below1pp=sum(abs(y-x)<.01 for x,y in flips),difference_over5pp=sum(abs(y-x)>=.05 for x,y in known),difference_over20pp=sum(abs(y-x)>=.2 for x,y in known),any_order_harm5pp=sum(x<-.05 or y<-.05 for x,y in known),any_order_harm20pp=sum(x<-.2 or y<-.2 for x,y in known))
    if not n:return out
    x,y=np.array(known).T;idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n))
    for k,v in [('order1',x),('order2',y),('order2_minus_order1',y-x),('absolute_order_difference',np.abs(y-x))]:out[k]=dict(mean=math.fsum(v)/n,**interval(np.sum(v[idx],axis=1)/n))
    out['flip_magnitudes']=dict(mean=math.fsum(abs(y-x) for x,y in flips)/len(flips) if flips else None,maximum=max(abs(y-x) for x,y in flips) if flips else None)
    z=np.array([abs(x)>EPS and abs(y)>EPS and x*y<0 for x,y in known],float);out['sign_flip_defined_pair_rate']=dict(value=math.fsum(z)/n,**interval(np.sum(z[idx],axis=1)/n))
    return out
def associations(rr,xfield,yfield):
    vals=[(r[xfield],r[yfield]) for r in rr if r.get(xfield) is not None and r.get(yfield) is not None];n=len(vals)
    if n<3:return dict(parents=n,Pearson=None,Spearman=None)
    x,y=np.array(vals).T;idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n));out=dict(parents=n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for method in ['Pearson','Spearman']:
            point=float(pearsonr(x,y).statistic if method=='Pearson' else spearmanr(x,y).statistic)
            xx=x[idx];yy=y[idx]
            if method=='Spearman':xx=rankdata(xx,axis=1);yy=rankdata(yy,axis=1)
            v=pearsonr(xx,yy,axis=1).statistic
            out[method]=dict(value=point if math.isfinite(point) else None,**interval(v))
    return out
def independent_summary(rr,summary):
    assert len(rr)==512 and len({r['parent_id'] for r in rr})==256;checks=0
    for r in rr:
        assert r['same_evidence'] and r['common_random_numbers_exact'] and r['query_residuals_both_exact_zero']
        assert abs(r['LN_difference_RMS']*math.sqrt(1536)-r['LN_difference_L2'])<1e-12
        for o in ['own_U_delta','common_U_delta']:
            a=r['order1_'+o];b=r['order2_'+o];d=None if a is None or b is None else b-a
            checks+=near(d,r[o+'_difference'])+near(None if d is None else abs(d),r['absolute_'+o+'_difference'])
    for ds,scopes in summary.items():
        for scope,subsets in scopes.items():
            for sub,expected in subsets.items():
                chosen=[r for r in rr if r['dataset']==ds and r['scope']==scope and (sub=='all128' or r['previous_P6']==(sub=='previous_P6_32'))]
                actual=dict(parents=len(chosen),same_initial_state_pairs=sum(r['initial_states_identical'] for r in chosen),outcomes={o:effects(chosen,o) for o in ['common_U_delta','own_U_delta']},associations={o:dict(LN_RMS_vs_absolute_effect_difference=associations(chosen,'LN_difference_RMS','absolute_'+o+'_difference'),Before_teacher_agreement_difference_vs_effect_difference=associations(chosen,'before_training_agreement_difference',o+'_difference')) for o in ['common_U_delta','own_U_delta']})
                checks+=near(actual,expected)
    return checks
def run(root):
    p=Path(root)/'results/stvg_opd_state_conditional_actionability/2026-10-13'
    with gzip.open(p/'ROWS.jsonl.gz','rt') as f:rr=[json.loads(v) for v in f]
    if (p/'SUMMARY.json').exists():s=json.loads((p/'SUMMARY.json').read_text())
    else:
        with gzip.open(p/'SUMMARY.json.gz','rt') as f:s=json.load(f)
    checks=independent_summary(rr,s)
    return dict(status='pass',scope='all512 anonymous parent-scope pairs/both orders/all128-old32-added96/full transitions and meaningful tails/independent scipy correlations and bootstrap',parents=256,parent_scope_pairs=512,independent_scalar_checks=checks,CPU_only=True,GT_read=False)
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),allow_nan=False))
