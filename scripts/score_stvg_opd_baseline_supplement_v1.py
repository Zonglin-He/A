"""CPU-only development selection and selected formal scoring, after barriers."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import gzip
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.stvg_opd_baseline_supplement_common_v1 import *

def rows(path):
    with gzip.open(path,'rt') as f: return [json.loads(x) for x in f]

def metrics(part):
    groups=collections.defaultdict(list)
    for r in part: groups[r['source_id']].append(r)
    fields=['After_v','Before_v','After_s','After_t','After_R30','After_R50','delta_v']
    ids=sorted(groups);mat=np.array([[np.mean([r[k] for r in groups[i]]) for k in fields] for i in ids])
    rng=np.random.default_rng(20261011); boots=np.concatenate([mat[rng.integers(0,len(ids),(100,len(ids)))].mean(1) for _ in range(100)])
    ci=np.quantile(boots,[.025,.975],axis=0)
    return dict(parents=len(ids),cells=len(part),values={k:dict(parent_macro=float(mat[:,j].mean()),query_macro=float(np.mean([r[k] for r in part])),ci95=ci[:,j].tolist()) for j,k in enumerate(fields)},
        harm_gt20pp_parents=int((mat[:,fields.index('delta_v')]<-.2).sum()),actual_update_arrivals=sum(r['updates']>0 for r in part),optimizer_steps=sum(r['updates'] for r in part),backward_steps=sum(r['backwards'] for r in part),skip_steps=sum(r['skips'] for r in part),recoveries=sum(r['recoveries'] for r in part),update_rate=sum(r['updates']>0 for r in part)/len(part),
        entropy_ratio_median=float(np.median([v for r in part for v in r['entropy_ratios']])),entropy_ratio_p95=float(np.quantile([v for r in part for v in r['entropy_ratios']],.95)))

def checked(phase):
    b=read(BASE/(phase.upper()+'_GLOBAL_BARRIER.json')); assert b['status']=='sealed' and not b['GT_read']
    for rel,h in b['barriers'].items():
        assert sha(ROOT/rel)==h; z=read(ROOT/rel)
        if z.get('reuse_identical_default'): continue
        for p,h2 in z['files'].items():
            rc=read((ROOT/p).with_suffix('.json')); assert sha(ROOT/p)==rc['sha256']==h2 and not rc['GT_read'] and rc['time']<=z['time']<=b['time']
    return b

def score(phase):
    verify();activate_old(); barrier=checked(phase)
    import torch
    torch.set_num_threads(2)
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    results={};counts=dict(arrivals=0,optimizer_scalar_checks=0,native_entropy_closures=0,state_chain_pairs=0,dense_scalar_checks=0)
    choices=None if phase=='development' else read(BASE/'SELECTED_CONFIGS.json')['selected']
    if not (BASE/(phase.upper()+'_GT_EXPOSURE.json')).exists(): write(BASE/(phase.upper()+'_GT_EXPOSURE.json'),dict(status='post_global_seal_only',barrier_sha256=sha(BASE/(phase.upper()+'_GLOBAL_BARRIER.json')),GT_read=True,time=time.time()))
    for ds,s in design()['datasets'].items():
        plan=read(PAPER/ds/'PLAN.json');sourceids={x:i for i,x in enumerate(sorted({r['source'] for r in plan['rows']}))}
        expected=design()['candidates'] if phase=='development' else {choices[ds][m]['candidate']:design()['candidates'][choices[ds][m]['candidate']] for m in ['TENT','SAR']}
        stage=dict(dataset=ds,source=s['source'],orders=s['development_orders'] if phase=='development' else plan['orders'],conditions=['clean'],parents=sorted({q for seq in (s['development_orders'] if phase=='development' else plan['orders']).values() for q in seq}))
        gt,spans,provenance=truths(ds,stage);results[ds]={}
        for candidate,definition in expected.items():
            dest=BASE/phase/ds/candidate; done=dest/'CPU_COMPLETION.json';out=PUB/phase/ds/candidate/'ROWS.jsonl.gz'
            if phase=='formal' and read(dest/'PREDICTION_BARRIER.json').get('reuse_identical_default') and not done.exists():
                from scripts.continue_stvg_opd_baseline_supplement_v1 import convert_default_rows
                convert_default_rows(ds,candidate)
            if done.exists():
                rc=read(done);assert sha(out)==rc['rows_sha256']; part=rows(out)
                for k,v in rc['counts'].items(): counts[k]+=v
            else:
                b=read(dest/'PREDICTION_BARRIER.json');part=[];local={k:0 for k in counts}; assert not b.get('reuse_identical_default'),'reuse uses original scalar export separately'
                for order,seq in b['orders'].items():
                    previous=None;state=None
                    for at,q in enumerate(seq):
                        p=dest/order/f'{at:05}.pt';z=load(p);assert sha(p)==b['files'][str(p.relative_to(ROOT))]
                        assert z['config']==definition['config'] and z['query_ordinal']==q and z['order']==order and z['arrival']==at and z['previous_payload_sha256']==previous
                        assert audit_baseline(z)==z['math_audit']
                        if state is not None:
                            from scripts.run_stvg_opd_baseline_supplement_v1 import equal_state
                            equal_state(state,z['state_before']);local['state_chain_pairs']+=1
                        assert z['state_before']['arrivals']==at and z['state_after']['arrivals']==at+1
                        state=z['state_after'];previous=sha(p); row=plan['rows'][q];values={}
                        for label in ['Before','After']:
                            pred=z[label];boxes=pred['boxes'].numpy();interval=pred['physical_interval']
                            val=official(boxes,row,gt[q],spans[q],interval,ds);ind=DenseTube(boxes,row,gt[q],spans[q],clip=ds=='hc2').score(interval)
                            for k in ['v','s','t']: assert abs(val[k]-ind[k])<2e-10;local['dense_scalar_checks']+=1
                            values[label]=val
                        audit=z['audit'];ratios=[];skips=0
                        for t in audit['trace']:
                            ceiling=t.get('margin',0)/definition['config'].get('margin_fraction',1) if 'margin' in t else sum(2*np.log(int(o['valid'].sum()))+np.log(2) for o in z['closure_evidence'][0])/len(z['closure_evidence'][0])
                            ratios.append(t['entropy']/ceiling);skips+=int('skip' in t)
                        r=dict(dataset=ds,method=definition['method'],candidate=candidate,config=definition['config'],source_id=sourceids[row['source']],query_ordinal=q,order=order,arrival=at,condition='clean',is_development_parent=row['source'] in s['excluded_primary_parent_sources'],updates=audit['counts']['optimizer_steps'],backwards=audit['counts']['backward_steps'],recoveries=audit['counts']['recoveries'],skips=skips,entropy_ratios=ratios,compute=z['compute'])
                        for label,val in values.items():
                            for k,v in val.items(): r[label+'_'+k]=v
                            r[label+'_R30']=float(val['v']>.3);r[label+'_R50']=float(val['v']>.5)
                        r['delta_v']=r['After_v']-r['Before_v'];part.append(r)
                        local['arrivals']+=1;local['optimizer_scalar_checks']+=z['math_audit']['optimizer']['scalar_checks'];local['native_entropy_closures']+=z['math_audit']['native_entropy_closures']
                        if at%100==0: print('BASELINE_SUPPLEMENT_CPU',phase,ds,candidate,order,at,flush=True)
                out.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(out,'wt') as f:
                    for r in part:f.write(json.dumps(r,allow_nan=False)+'\n')
                write(done,dict(status='complete_postseal_state_entropy_optimizer_dense',counts=local,rows_sha256=sha(out),GT_provenance=provenance,time=time.time()))
                for k,v in local.items():counts[k]+=v
            results[ds][candidate]=dict(all_population=metrics(part),primary=metrics([r for r in part if not r['is_development_parent']]) if phase=='formal' else None,rows_sha256=sha(out))
    write(PUB/(phase.upper()+'_STATISTICS.json'),dict(status='complete_postseal',statistics=results,counts=counts,all_negative_rows_retained=True,historically_exposed=True,bootstrap=10000,time=time.time()))
    if phase=='development':
        selected={}
        for ds,z in results.items():
            selected[ds]={}
            for method in ['TENT','SAR']:
                candidates=[(name,stat['all_population']) for name,stat in z.items() if design()['candidates'][name]['method']==method]
                best=max(s['values']['After_v']['parent_macro'] for _,s in candidates)
                tied=[(n,s) for n,s in candidates if best-s['values']['After_v']['parent_macro']<=1e-12]
                name,stat=min(tied,key=lambda x:(x[1]['harm_gt20pp_parents'],design()['candidates'][x[0]]['config']['lr'],design()['candidates'][x[0]]['config'].get('margin_fraction',0)))
                from vg_tta.decota_paper_baselines_20261005_v1 import CONFIGS
                cfg=design()['candidates'][name]['config']; selected[ds][method]=dict(candidate=name,config=cfg,dev_score=stat['values']['After_v']['parent_macro'],actual_update_rate=stat['update_rate'],reuse_identical_default=cfg==CONFIGS[method])
        write(BASE/'SELECTED_CONFIGS.json',dict(status='selected_only_from_prespecified_development',selected=selected,development_statistics_sha256=sha(PUB/'DEVELOPMENT_STATISTICS.json'),formal_GT_used_for_selection=False,time=time.time()))
    write(BASE/(phase.upper()+'_CPU_COMPLETION.json'),dict(status='complete_dev_selection' if phase=='development' else 'pending_actual_root_figures_publication',counts=counts,statistics_sha256=sha(PUB/(phase.upper()+'_STATISTICS.json')),experiment_complete=False,time=time.time()))

if __name__=='__main__': score(sys.argv[1])
