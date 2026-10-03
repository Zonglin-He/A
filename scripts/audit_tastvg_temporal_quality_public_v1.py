"""Standalone anonymous arithmetic audit; requires no features, GT or torch."""
import json,sys,collections
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())

def audit(folder):
    folder=Path(folder);checks=collections.Counter();largest=0.;allrows=[]
    def equal(a,b):
        nonlocal largest
        if isinstance(a,dict):
            assert set(a)==set(b),(set(a)-set(b),set(b)-set(a))
            for k in a:equal(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        elif isinstance(a,(float,int)) and not isinstance(a,bool):
            e=abs(float(a)-float(b));assert e<=1e-10,(a,b,e);largest=max(largest,e);checks['scalars']+=1
        else:assert a==b,(a,b)
    def grouped_summary(rows,fields):
        if not rows:return dict(sources=0,cells=0,metrics={})
        source_ids=sorted({r['source_id'] for r in rows});order_ids=sorted({r['order'] for r in rows})
        per={};order_means=collections.defaultdict(list)
        for s in source_ids:
            so=[]
            for order in order_ids:
                rr=[r for r in rows if r['source_id']==s and r['order']==order]
                if not rr:continue
                cc=[]
                for cond in sorted({r['condition'] for r in rr}):
                    z=[r for r in rr if r['condition']==cond];cc.append(np.mean([[r[f] for f in fields] for r in z],0))
                av=np.mean(cc,0);so.append(av);order_means[order].append(av)
            per[s]=np.mean(so,0)
        matrix=np.array([per[s] for s in source_ids]);rng=np.random.default_rng(20261003)
        boot=np.concatenate([matrix[rng.integers(0,len(matrix),(100,len(matrix)))].mean(1) for _ in range(100)])
        ci=np.percentile(boot,[2.5,97.5],axis=0);ov=np.array([np.mean(order_means[o],0) for o in order_ids])
        metrics={}
        for j,f in enumerate(fields):
            a=matrix[:,j];loo=(a.sum()-a)/(len(a)-1) if len(a)>1 else a
            metrics[f]=dict(mean=float(a.mean()),ci95=ci[:,j].tolist(),order_values=ov[:,j].tolist(),
                order_sample_SD=float(ov[:,j].std(ddof=1)) if len(ov)>1 else None,
                source_values={str(s):float(v) for s,v in zip(source_ids,a)},
                leave_one_out_range=[float(loo.min()),float(loo.max())],
                largest_influence_source=int(source_ids[int(np.argmax(abs(loo-a.mean())))]),
                cell_macro=float(np.mean([r[f] for r in rows])))
        return dict(sources=len(source_ids),cells=len(rows),metrics=metrics,bootstrap_draws=10000,seed=20261003)
    def pair(scores,values):
        wins=[]
        for i in range(8):
            for j in range(i+1,8):
                if abs(values[i]-values[j])<=1e-12:continue
                d=(scores[i]-scores[j])*np.sign(values[i]-values[j]);wins.append(1. if d>1e-12 else .5 if abs(d)<=1e-12 else 0.)
        return float(np.mean(wins)) if wins else 0.,len(wins)
    for split in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            rows=read(folder/split/ds/'ROWS.json');out=read(folder/split/ds/'SUMMARY.json');allrows.extend(rows)
            assert len(rows)==(384 if split=='search' else 192)
            for r in rows:
                equal(r['actual_gain'],r['quality_v']-r['A_v']);equal(r['actual_t_gain'],r['quality_t']-r['A_t'])
                equal(r['gross_gain'],max(r['actual_gain'],0));equal(r['gross_loss'],max(-r['actual_gain'],0))
                if not r['expert_scheduled']:
                    assert r['actual_gain']==r['actual_t_gain']==0;checks['identical_nonexpert']+=1;continue
                p=r['candidate_v'];t=r['candidate_t'];oi=r['old_selected'];qi=r['quality_selected']
                assert len(p)==len(t)==len(r['candidate_indices'])==8 and int(np.argmax(r['old_scores']))==oi
                for field,value in dict(A_v=p[oi],quality_v=p[qi],A_t=t[oi],quality_t=t[qi],native_v=p[0],native_t=t[0],
                    oracle_v=max(p),oracle_t=max(t),old_regret=max(p)-p[oi],quality_regret=max(p)-p[qi],
                    regret_reduction=p[qi]-p[oi],old_t_regret=max(t)-t[oi],quality_t_regret=max(t)-t[qi],
                    t_regret_reduction=t[qi]-t[oi],old_fast_gain=p[oi]-p[0],quality_fast_gain=p[qi]-p[0],
                    candidate_unique=len({tuple(x) for x in r['candidate_indices']}),interval_changed=int(qi!=oi)).items():equal(r[field],value)
                oldpa,n=pair(r['old_scores'],p);qpa,_=pair(r['quality_scores'],p)
                equal(r['strict_v_pairs'],n);equal(r['old_pair_v_accuracy'],oldpa);equal(r['quality_pair_v_accuracy'],qpa)
                equal(r['pair_v_accuracy_gain'],qpa-oldpa)
                # Alternative integral implementation: explicitly split each band
                # at each feature edge and integrate the constant signal.
                curve=r['activation_curve'];edges=r['feature_edges'];scores=[];details=[]
                def area(lo,hi):
                    value=0.;length=0.
                    for i,z in enumerate(curve):
                        s=max(lo,edges[i]);e=min(hi,edges[i+1])
                        if e>s:value+=z*(e-s);length+=e-s
                    return value,length
                global_value,_=area(0,1)
                for a,b in r['intervals_normalized']:
                    l=max(0,a-.25*(b-a));h=min(1,b+.25*(b-a))
                    iv,il=area(a,b);lv,ll=area(l,a);rv,rl=area(b,h)
                    inner=iv/il;outer=(lv+rv)/(ll+rl) if ll+rl>0 else global_value
                    z=dict(score=inner-outer,inner_mean=inner,outer_mean=outer,
                        left_mean=lv/ll if ll else None,right_mean=rv/rl if rl else None,
                        inside_length=il,outside_length=ll+rl,outer_window=[l,h],no_outer_observation=ll+rl==0)
                    details.append(z);scores.append(z['score'])
                equal(scores,r['quality_scores']);equal(details,r['quality_details'])
                if not r['embedding_available']:chosen=oi;reason='unavailable_embedding'
                elif max(curve)-min(curve)<=1e-12:chosen=oi;reason='constant_semantic_evidence'
                else:chosen=next(i for i,z in enumerate(scores) if max(scores)-z<=1e-12);reason='none'
                assert chosen==qi and reason==r['quality_fallback_reason']
                equal(r['activation_range'],max(curve)-min(curve))
                equal(r['quality_no_outer_candidates'],sum(d['no_outer_observation'] for d in details))
                checks['quality_decisions']+=1
            for group in ['corruption','clean']:
                for subset in ['all','expert','nonexpert']:
                    rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and
                        (subset=='all' or r['expert_scheduled']==(subset=='expert'))]
                    z=out[group][subset];ss=grouped_summary(rr,list(z['metrics']))
                    equal({k:z[k] for k in ss},ss)
                    cnt=dict(improved=sum(r['actual_gain']>1e-12 for r in rr),harmed=sum(r['actual_gain']<-1e-12 for r in rr),
                        unchanged=sum(abs(r['actual_gain'])<=1e-12 for r in rr),severe_harm_gt5pp=sum(r['actual_gain']<-.05 for r in rr),
                        baseline_good_gt03=sum(r['A_v']>.3 for r in rr),baseline_good_gt05=sum(r['A_v']>.5 for r in rr))
                    if subset=='expert':
                        cnt.update(old_rerank_gain_destroyed=sum(r['old_fast_gain']>1e-12 and r['quality_v']<r['A_v']-1e-12 for r in rr),
                            old_positive_rerank=sum(r['old_fast_gain']>1e-12 for r in rr),
                            replacement_v_below_native=sum(r['quality_selected']!=0 and r['quality_v']<r['native_v']-1e-12 for r in rr),
                            replacement_t_below_native=sum(r['quality_selected']!=0 and r['quality_t']<r['native_t']-1e-12 for r in rr),
                            fallback=sum(r['quality_fallback_reason']!='none' for r in rr))
                        den=ss['metrics']['old_regret']['mean'];num=ss['metrics']['regret_reduction']['mean']
                        equal(z['regret_recovered_fraction'],num/den if den>1e-12 else None)
                        valid=[r for r in rr if r['strict_v_pairs']>0]
                        px=grouped_summary(valid,list(z['pairwise_v']['metrics']));px['no_strict_pair_cells']=len(rr)-len(valid)
                        equal(z['pairwise_v'],px)
                    equal(z['counts'],cnt)
                    equal(z['correctness'],{str(th):dict(correct_to_wrong=sum(r['A_v']>th and r['quality_v']<=th for r in rr),
                        wrong_to_correct=sum(r['A_v']<=th and r['quality_v']>th for r in rr)) for th in [.3,.5]})
                    checks['summary_groups']+=1
    assert len(allrows)==1152 and checks['quality_decisions']==288 and checks['identical_nonexpert']==864
    assert len({(r['dataset'],r['split'],r['condition'],r['order'],r['arrival']) for r in allrows})==1152
    return dict(status='pass',checks=dict(checks),max_arithmetic_error=largest,GT_features_weights_not_required=True)

if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),ensure_ascii=False,indent=2))
