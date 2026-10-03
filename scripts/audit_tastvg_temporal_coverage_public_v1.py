"""Standalone anonymous arithmetic, decision and paired-bootstrap audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,collections
from pathlib import Path
import numpy as np

def run(path):
    p=Path(path);counts=collections.Counter()
    def read(f):return json.loads(f.read_text())
    def close(a,b):
        aa,bb=np.asarray(a,float),np.asarray(b,float)
        assert aa.shape==bb.shape and np.allclose(aa,bb,rtol=0,atol=1e-10),(a,b)
        counts['scalar_checks']+=aa.size
    def summary(rows,z):
        assert len(rows)==z['cells'] and len({r['source_id'] for r in rows})==z['sources']
        fields=list(z['metrics']);ids=sorted({r['source_id'] for r in rows});mat=[]
        order_values={o:[] for o in sorted({r['order'] for r in rows})}
        for source in ids:
            src=[]
            for order in order_values:
                rc=[r for r in rows if r['source_id']==source and r['order']==order]
                if not rc:continue
                cond=[]
                for c in sorted({r['condition'] for r in rc}):
                    cond.append(np.mean([[r[f] for f in fields] for r in rc if r['condition']==c],axis=0))
                value=np.mean(cond,axis=0);src.append(value);order_values[order].append(value)
            mat.append(np.mean(src,axis=0))
        mat=np.array(mat);rng=np.random.default_rng(20261003)
        b=np.concatenate([mat[rng.integers(0,len(mat),(100,len(mat)))].mean(1) for _ in range(100)])
        ci=np.quantile(b,[.025,.975],axis=0)
        ov=np.array([np.mean(v,axis=0) for v in order_values.values()])
        for j,f in enumerate(fields):
            m=z['metrics'][f];a=mat[:,j];loo=(a.sum()-a)/(len(a)-1) if len(a)>1 else a
            close(a.mean(),m['mean']);close(ci[:,j],m['ci95']);close(ov[:,j],m['order_values'])
            close([loo.min(),loo.max()],m['leave_one_out_range'])
            close(a,[m['source_values'][str(s)] for s in ids]);close(np.mean([r[f] for r in rows]),m['cell_macro'])
            assert m['largest_influence_source']==ids[int(np.argmax(abs(loo-a.mean())))]
        assert z['counts']==dict(improved=sum(r['actual_gain']>1e-12 for r in rows),
            harmed=sum(r['actual_gain']<-1e-12 for r in rows),unchanged=sum(abs(r['actual_gain'])<=1e-12 for r in rows),
            severe_harm_gt5pp=sum(r['actual_gain']<-.05 for r in rows))
        for th in [.3,.5]:
            assert z['correctness'][str(th)]==dict(correct_to_wrong=sum(r['A_v']>th and r['new_v']<=th for r in rows),
                wrong_to_correct=sum(r['A_v']<=th and r['new_v']>th for r in rows))
    n=e=0
    for ds in ['vidstg','hc2']:
        for split in ['search','confirm']:
            rows=read(p/split/ds/'ROWS.json');z=read(p/split/ds/'SUMMARY.json')
            assert len(rows)==(384 if split=='search' else 192)
            for r in rows:
                close(r['actual_gain'],r['new_v']-r['A_v']);close(r['actual_t_gain'],r['new_t']-r['A_t'])
                close(r['gross_gain'],max(r['actual_gain'],0));close(r['gross_loss'],max(-r['actual_gain'],0))
                close(r['GT_time_minus_grid'],r['GT_time_v']-r['grid_oracle_v'])
                close(r['grid_headroom_vs_A'],r['grid_oracle_v']-r['A_v'])
                close(r['GT_headroom_vs_A'],r['GT_time_v']-r['A_v'])
                assert r['grid_pair_count']==r['sampled_frames']*(r['sampled_frames']-1)//2
                assert r['GT_time_v']>=r['grid_oracle_v']-1e-12>=max(r['A_v'],r['new_v'])-2e-12
                if r['expert_scheduled']:
                    for name in ['old','new']:
                        vv=r[name+'_candidate_v'];tt=r[name+'_candidate_t'];scores=r[name+'_scores'];k=r[name+'_selected']
                        inds=r[name+'_candidate_indices']
                        assert len(vv)==len(tt)==len(scores)==len(inds)==8
                        assert k==int(np.argmax(scores))
                        close(vv[k],r['A_v'] if name=='old' else r['new_v'])
                        close(tt[k],r['A_t'] if name=='old' else r['new_t'])
                        close(vv[0],r['native_v']);close(max(vv),r[name+'_oracle_v'])
                        close(r[name+'_selection_gap'],max(vv)-(r['A_v'] if name=='old' else r['new_v']))
                        close(r[name+'_coverage_gap'],r['GT_time_v']-max(vv))
                        close(r['grid_minus_'+name+'_oracle'],r['grid_oracle_v']-max(vv))
                        close(r[name+'_coverage_gap'],r['grid_minus_'+name+'_oracle']+r['GT_time_minus_grid'])
                        close(r[name+'_oracle_headroom'],max(vv)-r['native_v'])
                        assert len({tuple(q) for q in inds})==r[name+'_unique']
                    close(r['oracle_gain'],r['new_oracle_v']-r['old_oracle_v'])
                    close(r['old_fast_gain'],r['A_v']-r['native_v']);close(r['new_fast_gain'],r['new_v']-r['native_v'])
                    assert r['new_unique']==8 and r['old_candidate_indices'][0]==r['new_candidate_indices'][0]
                    assert r['pool_overlap']==len(set(map(tuple,r['old_candidate_indices']))&set(map(tuple,r['new_candidate_indices'])))
                    e+=1
                else:close(r['actual_gain'],0);close(r['actual_t_gain'],0)
                n+=1
            for group in ['corruption','clean']:
                for sub in ['all','expert','nonexpert']:
                    rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and
                        (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
                    summary(rr,z[group][sub])
    diagnostic=p/'SUPPORT_AND_SELECTOR_DIAGNOSIS.json'
    if diagnostic.exists():
        stored=read(diagnostic)
        for split in ['search','confirm']:
            for ds in ['vidstg','hc2']:
                rr=read(p/split/ds/'ROWS.json')
                for group in ['corruption','clean']:
                    rows=[r for r in rr if r['expert_scheduled'] and (r['condition']!='clean')==(group=='corruption')]
                    assert stored[f'{split}/{ds}/{group}']==diagnose(rows)
                    counts['support_selector_diagnostic_groups']+=1
    assert n==1152 and e==288
    result=dict(status='pass',arrivals=n,expert_pools=e,checks=dict(counts),
        GT_read=False,anonymous_only=True,new_model_calls=0,new_expert_calls=0)
    print(json.dumps(result,indent=2));return result

def diagnose(rows):
    def retained(r):return r['old_candidate_indices'][r['old_selected']] in r['new_candidate_indices']
    harmed=[r for r in rows if r['actual_gain']<-1e-12]
    return dict(cells=len(rows),old_A_in_new=sum(retained(r) for r in rows),harmed=len(harmed),
        new_pool_forced_harm=sum(r['new_oracle_v']<r['A_v']-1e-12 for r in rows),
        preservable_harm=sum(r['new_oracle_v']>=r['A_v']-1e-12 for r in harmed),
        old_A_retained_harm=sum(retained(r) for r in harmed),
        oracle_improved_actual_harm=sum(r['oracle_gain']>1e-12 for r in harmed),
        higher_critic_score_harm=sum(max(r['new_scores'])>max(r['old_scores'])+1e-12 for r in harmed))
if __name__=='__main__':run(sys.argv[1])
