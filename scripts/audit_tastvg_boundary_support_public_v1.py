"""Standalone independent support/transition/arithmetic/bootstrap public audit."""
import json,sys,collections
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())

def aggregate(rows,fields):
    if not rows:return dict(sources=0,cells=0,metrics={})
    sources=sorted({r['source_id'] for r in rows});orders=sorted({r['order'] for r in rows})
    per={};byorder=collections.defaultdict(list)
    for s in sources:
        vals=[]
        for o in orders:
            rr=[r for r in rows if r['source_id']==s and r['order']==o]
            if not rr:continue
            cc=[np.mean([[r[f] for f in fields] for r in rr if r['condition']==c],0)
                for c in sorted({r['condition'] for r in rr})]
            v=np.mean(cc,0);vals.append(v);byorder[o].append(v)
        per[s]=np.mean(vals,0)
    mat=np.array([per[s] for s in sources]);rng=np.random.default_rng(20261003)
    boot=np.concatenate([mat[rng.integers(0,len(mat),(100,len(mat)))].mean(1) for _ in range(100)])
    ci=np.percentile(boot,[2.5,97.5],0);ov=np.array([np.mean(byorder[o],0) for o in orders]);metrics={}
    for j,f in enumerate(fields):
        a=mat[:,j];loo=(a.sum()-a)/(len(a)-1) if len(a)>1 else a
        metrics[f]=dict(mean=float(a.mean()),ci95=ci[:,j].tolist(),order_values=ov[:,j].tolist(),
            order_sample_SD=float(ov[:,j].std(ddof=1)) if len(ov)>1 else None,
            source_values={str(s):float(v) for s,v in zip(sources,a)},leave_one_out_range=[float(loo.min()),float(loo.max())],
            largest_influence_source=int(sources[int(np.argmax(abs(loo-a.mean())))]),
            cell_macro=float(np.mean([r[f] for r in rows])))
    return dict(sources=len(sources),cells=len(rows),metrics=metrics,bootstrap_draws=10000,seed=20261003)

def independent_append(start,end,old):
    # Alternative implementation computes min distances one seed at a time.
    pairs=[(i,j) for i in range(len(start)) for j in range(i+1,len(start))]
    xy=np.array([[start[i],end[j]] for i,j in pairs]);selected=[tuple(x) for x in old]
    while len(selected)<32:
        best=np.full(len(pairs),np.inf)
        for a,b in selected:
            d=(xy[:,0]-start[a])**2+(xy[:,1]-end[b])**2;best=np.minimum(best,d)
        for t in selected:best[pairs.index(t)]=-1.
        selected.append(pairs[int(np.argmax(best))])
    return [list(x) for x in selected]

def pairwise(scores,vals):
    wins=[]
    for i in range(len(vals)):
        for j in range(i+1,len(vals)):
            if abs(vals[i]-vals[j])<=1e-12:continue
            d=(scores[i]-scores[j])*np.sign(vals[i]-vals[j])
            wins.append(1. if d>1e-12 else .5 if abs(d)<=1e-12 else 0.)
    return float(np.mean(wins)) if wins else 0.,len(wins)

def audit(folder):
    folder=Path(folder);cnt=collections.Counter();maxerr=0.;arms=['A8','B8','D8','A32','B32','D32'];allrows=[]
    diffs=dict(A_support=('A32','A8'),B_support=('B32','B8'),D_support=('D32','D8'),
        B8_vs_A8=('B8','A8'),D8_vs_A8=('D8','A8'),B32_vs_A32=('B32','A32'),
        D32_vs_A32=('D32','A32'),D32_vs_D8=('D32','D8'))
    def eq(a,b):
        nonlocal maxerr
        if isinstance(a,dict):
            assert set(a)==set(b),(set(a)-set(b),set(b)-set(a))
            for k in a:eq(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):eq(x,y)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            er=abs(float(a)-float(b));assert er<1e-10,(a,b,er);maxerr=max(maxerr,er);cnt['scalar_arithmetic']+=1
        else:assert a==b,(a,b)
    for sp in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            rows=read(folder/sp/ds/'ROWS.json');out=read(folder/sp/ds/'SUMMARY.json');allrows.extend(rows)
            assert len(rows)==(384 if sp=='search' else 192)
            for r in rows:
                for arm in arms:
                    eq(r[f'{arm}_gain'],r[f'{arm}_v']-r['A8_v']);eq(r[f'{arm}_t_gain'],r[f'{arm}_t']-r['A8_t'])
                    eq(r[f'{arm}_gross_gain'],max(r[f'{arm}_gain'],0));eq(r[f'{arm}_gross_loss'],max(-r[f'{arm}_gain'],0))
                for name,(a,b) in diffs.items():
                    for m in ['v','t']:eq(r[f'{name}_{m}'],r[f'{a}_{m}']-r[f'{b}_{m}'])
                if not r['expert_scheduled']:
                    assert all(r[f'{a}_gain']==r[f'{a}_t_gain']==0 for a in arms);cnt['nonexpert_identical']+=1;continue
                assert len(r['candidate_v'])==len(r['candidate_t'])==len(r['candidate_indices'])==32
                assert independent_append(r['grid_start'],r['grid_end'],r['candidate_indices'][:8])==r['candidate_indices']
                cnt['independent_expansion']+=1
                pairs=r['candidate_indices']
                eq([[r['grid_start'][i],r['grid_end'][j]] for i,j in pairs],r['intervals_normalized'])
                curve=r['activation_curve'];edges=r['feature_edges']
                def area(lo,hi):
                    val=0.;length=0.
                    for i,z in enumerate(curve):
                        a=max(lo,edges[i]);b=min(hi,edges[i+1])
                        if b>a:val+=z*(b-a);length+=b-a
                    return val,length
                for n in [8,32]:
                    for m in ['v','t']:
                        vals=r[f'candidate_{m}'][:n];eq(r[f'O{n}_{m}'],max(vals))
                        for a in ['A','B','D']:
                            arm=f'{a}{n}';at=r['choices'][arm]
                            assert 0<=at<n and len(r['scores'][arm])==n
                            eq(r[f'{arm}_{m}'],vals[at]);eq(r[f'{arm}_{"regret" if m=="v" else "t_regret"}'],max(vals)-vals[at])
                            p,den=pairwise(r['scores'][arm],vals);eq(r[f'{arm}_pair_{m}'],p);eq(r[f'strict_{n}_{m}_pairs'],den)
                    assert int(np.argmax(r['scores'][f'A{n}']))==r['choices'][f'A{n}']
                    bd=[];dd=[]
                    for a,b in r['intervals_normalized'][:n]:
                        iv,il=area(a,b);l=max(0.,a-.25*(b-a));h=min(1.,b+.25*(b-a))
                        lv,ll=area(l,a);rv,rl=area(b,h);gv,_=area(0,1)
                        inner=iv/il;outer=(lv+rv)/(ll+rl) if ll+rl else gv
                        bd.append(dict(score=inner-outer,inner_mean=inner,outer_mean=outer,left_mean=lv/ll if ll else None,
                            right_mean=rv/rl if rl else None,inside_length=il,outside_length=ll+rl,
                            outer_window=[l,h],no_outer_observation=ll+rl==0))
                        w=1./r['duration_seconds'];ranges=[[a,min(b,a+w)],[max(0.,a-w),a],[max(a,b-w),b],[b,min(1.,b+w)]]
                        vv=[area(x,y) for x,y in ranges];mm=[v/z if z else None for v,z in vv]
                        ss=mm[0]-mm[1] if mm[1] is not None else 0.;ee=mm[2]-mm[3] if mm[3] is not None else 0.
                        dd.append(dict(score=min(ss,ee),start_transition=ss,end_transition=ee,means=mm,ranges=ranges,
                            lengths=[z for _,z in vv],missing_start_context=mm[1] is None,missing_end_context=mm[3] is None,
                            both_positive=ss>0 and ee>0))
                    for i,d in enumerate(dd):
                        saved=r['details'][f'D{n}'][i]
                        # Integral floats are compared below. A strict sign flag
                        # is audited against the saved float operands: alternate
                        # summation can flip the sign of an exact-zero transition.
                        flag=saved['start_transition']>0 and saved['end_transition']>0
                        assert flag==saved['both_positive']
                        if d['both_positive']!=flag:
                            assert min(abs(d['start_transition']),abs(d['end_transition']))<=1e-12
                            cnt['near_zero_diagnostic_sign_rounding']+=1
                        d['both_positive']=flag
                    for a,details in [('B',bd),('D',dd)]:
                        arm=f'{a}{n}';eq(details,r['details'][arm]);sc=[d['score'] for d in details];eq(sc,r['scores'][arm])
                        if not r['embedding_available']:chosen=r['choices'][f'A{n}'];reason='unavailable_embedding'
                        elif max(curve)-min(curve)<=1e-12:chosen=r['choices'][f'A{n}'];reason='constant_semantic_evidence'
                        else:chosen=next(i for i,s in enumerate(sc) if max(sc)-s<=1e-12);reason='none'
                        assert chosen==r['choices'][arm] and reason==r['fallbacks'][arm]
                        cnt['independent_B_D_decisions']+=1
                    eq(r[f'D{n}_missing_candidates'],sum(d['missing_start_context'] or d['missing_end_context'] for d in dd))
                    eq(r[f'D{n}_selected_both_positive'],int(dd[r['choices'][f'D{n}']]['both_positive']))
                eq(r['capacity_gain'],r['O32_v']-r['O8_v']);assert r['capacity_gain']>=0
                eq(r['t_capacity_gain'],r['O32_t']-r['O8_t']);eq(r['old_fast_gain'],r['A8_v']-r['candidate_v'][0])
                eq(r['activation_range'],max(curve)-min(curve))
                for arm in arms:eq(r[f'{arm}_changed'],int(r['choices'][arm]!=r['choices']['A8']))
            for group in ['corruption','clean']:
                for sub in ['all','expert','nonexpert']:
                    rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and
                        (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
                    z=out[group][sub];s=aggregate(rr,list(z['metrics']));eq({k:z[k] for k in s},s)
                    for arm in arms:
                        g=f'{arm}_gain';az=dict(improved=sum(r[g]>1e-12 for r in rr),harmed=sum(r[g]<-1e-12 for r in rr),
                            unchanged=sum(abs(r[g])<=1e-12 for r in rr),severe_harm_gt5pp=sum(r[g]<-.05 for r in rr),
                            correctness={str(t):dict(correct_to_wrong=sum(r['A8_v']>t and r[f'{arm}_v']<=t for r in rr),
                                wrong_to_correct=sum(r['A8_v']<=t and r[f'{arm}_v']>t for r in rr)) for t in [.3,.5]})
                        if sub=='expert':az.update(old_positive_fast=sum(r['old_fast_gain']>1e-12 for r in rr),
                            old_positive_fast_destroyed=sum(r['old_fast_gain']>1e-12 and r[g]<-1e-12 for r in rr))
                        eq(az,z['arms'][arm])
                    eq(z['comparisons'],{name:dict(improved=sum(r[f'{name}_v']>1e-12 for r in rr),
                        harmed=sum(r[f'{name}_v']<-1e-12 for r in rr),unchanged=sum(abs(r[f'{name}_v'])<=1e-12 for r in rr),
                        severe_harm_gt5pp=sum(r[f'{name}_v']<-.05 for r in rr)) for name in diffs})
                    if sub=='expert':
                        for n in [8,32]:
                            for m in ['v','t']:
                                valid=[r for r in rr if r[f'strict_{n}_{m}_pairs']>0]
                                pz=aggregate(valid,[f'{a}{n}_pair_{m}' for a in ['A','B','D']]);pz['no_strict_pair_cells']=len(rr)-len(valid)
                                eq(pz,z['pairwise'][f'{n}_{m}'])
                    cnt['summary_groups']+=1
    assert len(allrows)==1152 and cnt['nonexpert_identical']==864 and cnt['independent_expansion']==288
    assert len({(r['dataset'],r['split'],r['condition'],r['order'],r['arrival']) for r in allrows})==1152
    return dict(status='pass',checks=dict(cnt),max_arithmetic_error=maxerr,GT_features_weights_not_required=True)

if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2))
