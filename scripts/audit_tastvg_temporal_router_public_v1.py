"""Recompute anonymous score rules, transitions and source bootstrap, CPU only.

Does not require media, proposals' coordinates, GT boxes, states or a model.
The raw interval/mask equivalence audit is separately recorded in ROOT_READBACK.
"""
import sys,json,collections
from pathlib import Path
import numpy as np
N_CHECKS=0
def read(p):return json.loads(Path(p).read_text())
def equal(a,b):
    global N_CHECKS
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(set(a),set(b))
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(float,int,np.floating,np.integer)) and not isinstance(a,bool):
        assert b is not None and abs(float(a)-float(b))<2e-10,(a,b);N_CHECKS+=1
    else:assert a==b,(a,b);N_CHECKS+=1

def macro(rows,fields):
    requested=len(rows);rows=[r for r in rows if all(r.get(k) is not None for k in fields)]
    if not rows:return dict(sources=0,cells=0,metrics={},requested_cells=requested,unavailable_cells=requested)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['source_id'],r['order'],r['condition']].append([r[k] for k in fields])
    so=collections.defaultdict(list)
    for (s,o,c),v in groups.items():so[s,o].append(np.mean(v,axis=0))
    ss=collections.defaultdict(list);orders=collections.defaultdict(list)
    for (s,o),v in so.items():z=np.mean(v,axis=0);ss[s].append(z);orders[o].append(z)
    mat=np.asarray([np.mean(ss[s],axis=0) for s in sorted(ss)]);rng=np.random.default_rng(20261001)
    bootstrap=np.concatenate([mat[rng.integers(0,len(mat),(100,len(mat)))].mean(1) for _ in range(100)])
    ci=np.percentile(bootstrap,[2.5,97.5],axis=0);ov=np.asarray([np.mean(orders[o],axis=0) for o in sorted(orders)])
    return dict(sources=len(mat),cells=len(rows),requested_cells=requested,unavailable_cells=requested-len(rows),
        metrics={k:dict(mean=float(mat[:,j].mean()),ci95=ci[:,j].tolist(),query_macro=float(np.mean([r[k] for r in rows])),
        order_values=ov[:,j].tolist(),order_sample_SD=float(ov[:,j].std(ddof=1)) if len(ov)>1 else None,
        harm_gt5pp_sources=int((mat[:,j]<-.05).sum()) if k.startswith('delta_') else None) for j,k in enumerate(fields)})

def subset(rows,g,s='all'):
    return [r for r in rows if (r['condition']!='clean')==(g=='corruption') and (s=='all' or r['expert_scheduled']==(s=='expert'))]

def changes(rows,a,b):
    out={}
    for m in ['t','v']:
        out[m]={}
        for t in [.3,.5]:
            before=[r[a+'_'+m]>t for r in rows];after=[r[b+'_'+m]>t for r in rows]
            out[m][str(t)]=dict(correct_before=sum(before),wrong_before=len(rows)-sum(before),
                correct_to_wrong=sum(x and not y for x,y in zip(before,after)),
                wrong_to_correct=sum(not x and y for x,y in zip(before,after)),
                correct_unchanged=sum(x and y for x,y in zip(before,after)),
                wrong_unchanged=sum(not x and not y for x,y in zip(before,after)))
    return out

def run(base):
    global N_CHECKS
    base=Path(base);support_fields=['event_recall','gt_mass','scored_gt_mass','scored_recall','soft_iou','quantile_coverage','quantile_scored_coverage','uniform_coverage']
    task_fields=['Frozen_t','Frozen_v','Native_t','Native_v','Current_t','Current_v','QC_t','QC_v','delta_t','delta_v','delta_Current_Frozen_t','delta_Current_Frozen_v','delta_QC_Frozen_t','delta_QC_Frozen_v']
    signals=[];total=experts=0
    for ds in ['vidstg','hc2']:
        folder=base/ds;c=read(folder/'CONFIG.json');rows=read(folder/'ROWS.json');erows=read(folder/'EXPERT_ROWS.json')
        assert len(rows)==384 and len(erows)==96 and c['sources']==c['queries']==32
        assert len({r['source_id'] for r in rows})==32 and len({(r['condition'],r['order'],r['arrival']) for r in rows})==384
        ed={(r['condition'],r['order'],r['arrival']):r for r in erows}
        for r in rows:
            assert c['orders'][r['order']][r['arrival']]==r['parent'] and r['expert_scheduled']==(r['arrival']%4==0)
            for m in ['t','v']:
                equal(r['delta_'+m],r['QC_'+m]-r['Current_'+m])
                for name in ['Current','QC']:equal(r['delta_'+name+'_Frozen_'+m],r[name+'_'+m]-r['Frozen_'+m])
            if not r['expert_scheduled']:assert r['delta_t']==r['delta_v']==0. and r['Current_t']==r['Native_t'] and r['Current_v']==r['Native_v']
            else:
                e=ed[(r['condition'],r['order'],r['arrival'])]
                for k in r:equal(e[k],r[k])
        for r in erows:
            n=r['proposal_count'];k=r['candidate_count'];conf=np.asarray(r['proposal_confidence']);a=np.asarray(r['proposal_agreement']);q=np.asarray(r['proposal_quality'])
            match=np.asarray(r['candidate_proposal_overlap']).reshape(k,n)
            assert 0<k<=8 and conf.shape==a.shape==q.shape==(n,) and (conf>=0).all() and ((a>=0)&(a<=1+1e-12)).all()
            equal(a.tolist(),(np.asarray(r['proposal_overlap_row_sums'])/(n-1) if n>1 else np.zeros(n)).tolist());equal(q.tolist(),(conf*a).tolist())
            prop=np.asarray(r['proposal_gt_t'])
            for name,weights in [('Current',conf),('QC',q)]:
                score=(match*weights).max(1) if n else np.zeros(k);sel=int(np.argmax(score))
                equal(score.tolist(),r[name+'_scores']);assert sel==r[name+'_selected']
                con=match[sel]*weights;wi=int(np.argmax(con)) if n and con.max()>0 else None
                equal(wi,r[name+'_winning_contributor'])
                equal(bool(wi is not None and prop[wi]<=.3 and prop.max()>.5),r[name+'_bad_contributor'])
                equal(float(prop.max()-prop[wi]) if wi is not None else None,r[name+'_contributor_regret'])
                for m in ['t','v']:
                    equal(r[name+'_'+m],r['candidate_'+m][sel]);equal(r[name+'_regret_'+m],max(r['candidate_'+m])-r[name+'_'+m])
            for m in ['t','v']:equal(r['delta_regret_'+m],r['QC_regret_'+m]-r['Current_regret_'+m])
            equal(r['selection_changed'],r['Current_selected']!=r['QC_selected'])
            for name in ['S','E','SE']:
                st=r[name+'_statistics'];w=st['total_mass'];inn=st['event_mass'];gt=st['event_frames'];sg=st['scored_frames']
                vals=dict(gt_mass=inn/w if w else None,scored_gt_mass=st['scored_mass']/w if w else None,
                    event_recall=st['positive_event_frames']/gt if gt else None,scored_recall=st['positive_scored_frames']/sg if sg else None,
                    soft_iou=inn/(w+gt-inn) if w+gt-inn else None,
                    quantile_coverage=st['quantile_hits']/5 if st['quantile_hits'] is not None else None,
                    quantile_scored_coverage=st['quantile_scored_hits']/5 if st['quantile_scored_hits'] is not None else None,uniform_coverage=st['uniform_hits']/5)
                for key,v in vals.items():equal(v,st[key]);equal(v,r[name+'_'+key])
                assert 0<=inn<=min(w,gt)+1e-12 and st['raw_gt_references']<=st['raw_valid_references']<=5
                equal(st['available'],w>0);equal(st['useful_reference_discarded'],st['raw_gt_references']>0 and st['valid_reference_mass']==0)
            for key in ['total_mass','event_mass','scored_mass','valid_reference_mass']:equal(r['SE_statistics'][key],.5*(r['S_statistics'][key]+r['E_statistics'][key]))
        saved=read(folder/'SUPPORT_SUMMARY.json')
        for g in ['corruption','clean']:
            rr=subset(erows,g)
            for router in ['S','E','SE']:
                z=saved[g][router];v={k:val for k,val in z.items() if k!='counts'}
                equal(macro(rr,[router+'_'+k for k in support_fields]),v)
                counts=dict(cells=len(rr),available=sum(r[router+'_statistics']['available'] for r in rr),
                    missing_sampled_event=sum(r[router+'_statistics']['event_frames']==0 for r in rr),
                    any_quantile_gt=sum(r[router+'_statistics']['quantile_any_gt'] for r in rr),
                    quantile_duplicates=sum(r[router+'_statistics']['available'] and r[router+'_statistics']['quantile_unique']<5 for r in rr),
                    nonempty_reference_arrivals=sum(r[router+'_statistics']['raw_valid_references']>0 for r in rr),
                    zero_weighted_mass_nonempty=sum(r[router+'_statistics']['raw_valid_references']>0 and r[router+'_statistics']['valid_reference_mass']==0 for r in rr),
                    useful_reference_discarded=sum(r[router+'_statistics']['useful_reference_discarded'] for r in rr))
                equal(counts,z['counts'])
            for router in ['E','SE']:
                dr=[]
                for r in rr:
                    x={k:r[k] for k in ['source_id','order','condition']}
                    for k in support_fields:x['delta_'+k]=r[router+'_'+k]-r['S_'+k] if r[router+'_'+k] is not None and r['S_'+k] is not None else None
                    dr.append(x)
                equal(macro(dr,['delta_'+k for k in support_fields]),saved[g][router+'_minus_S'])
        ts=read(folder/'TASK_SUMMARY.json')
        for g in ['corruption','clean']:
            for sub in ['all','expert','nonexpert']:
                rr=subset(rows,g,sub);z=ts[g][sub];fields={k:z[k] for k in ['sources','cells','metrics','requested_cells','unavailable_cells']}
                equal(macro(rr,task_fields),fields);equal(changes(rr,'Current','QC'),z['Current_to_QC'])
                delta=[r['delta_v'] for r in rr]
                equal(np.mean([max(v,0) for v in delta])*100,z['cell_gross_gain_v_pp']);equal(-np.mean([min(v,0) for v in delta])*100,z['cell_gross_loss_v_pp'])
                equal(sum(v<-.05 for v in delta),z['severe_harm_gt5pp']);equal(sum(v>.05 for v in delta),z['severe_gain_gt5pp'])
                if sub=='expert':
                    equal(changes(rr,'Native','Current'),z['Native_to_Current']);equal(changes(rr,'Native','QC'),z['Native_to_QC'])
                    er=subset(erows,g)
                    equal(macro(er,['Current_regret_t','Current_regret_v','QC_regret_t','QC_regret_v','delta_regret_t','delta_regret_v']),z['selection_regret'])
                    for name in ['Current','QC']:
                        expected=dict(bad_contributor=sum(r[name+'_bad_contributor'] for r in er),
                            selected_harms_native_t=sum(r[name+'_t']<r['Native_t']-1e-12 for r in er),selected_harms_native_v=sum(r[name+'_v']<r['Native_v']-1e-12 for r in er),
                            missed_correct={m:{str(th):sum(max(r['candidate_'+m])>th and r[name+'_'+m]<=th for r in er) for th in [.3,.5]} for m in ['t','v']},
                            mean_contributor_regret=float(np.mean([r[name+'_contributor_regret'] for r in er if r[name+'_contributor_regret'] is not None])))
                        equal(expected,z['critic_diagnosis'][name])
        decision=read(folder/'DECISION.json');ct=ts['corruption']['expert'];checks={r:dict(mass_gain=saved['corruption'][r+'_minus_S']['metrics']['delta_gt_mass']['mean'],reference_gain=saved['corruption'][r+'_minus_S']['metrics']['delta_quantile_coverage']['mean'],recall_gain=saved['corruption'][r+'_minus_S']['metrics']['delta_event_recall']['mean']) for r in ['E','SE']}
        protected=all(ct['Native_to_QC'][m][t]['correct_to_wrong']<=ct['Native_to_Current'][m][t]['correct_to_wrong'] for m in ['t','v'] for t in ['0.3','0.5'])
        signal=any(v['mass_gain']>1e-12 and v['reference_gain']>1e-12 and v['recall_gain']>=-1e-12 for v in checks.values()) and ct['metrics']['delta_t']['mean']>1e-12 and protected
        equal(dict(T1_development_signal=signal,support_checks=checks,QC_mean_t_gain=ct['metrics']['delta_t']['mean'],native_correct_destruction_not_increased=protected,H_full_started=False,new_model_execution=False,spatial_A_changed=False),decision)
        if signal:signals.append(ds)
        total+=len(rows);experts+=len(erows)
    equal(read(base/'DECISION.json'),dict(status='T1_exact_CPU_readout_released' if signals else 'T0_completed_T1_not_triggered',T1_triggered=bool(signals),new_GPU_trial=False,H_full_started=False,production_promoted=False,signal_datasets=signals))
    result=dict(status='pass',scalar_checks=N_CHECKS,rows=total,expert_rows=experts,source_bootstrap_recomputed=True)
    print(json.dumps(result));return result

if __name__=='__main__':run(sys.argv[1])
