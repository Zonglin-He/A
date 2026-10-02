"""Post-seal CPU GT diagnosis and independent raw-input numerical readback."""
from tastvg_temporal_router_common_v1 import *
import numpy as np

SUPPORT_FIELDS=['event_recall','gt_mass','scored_gt_mass','scored_recall','soft_iou',
                'quantile_coverage','quantile_scored_coverage','uniform_coverage']
TASK_FIELDS=['Frozen_t','Frozen_v','Native_t','Native_v','Current_t','Current_v',
             'QC_t','QC_v','delta_t','delta_v','delta_Current_Frozen_t',
             'delta_Current_Frozen_v','delta_QC_Frozen_t','delta_QC_Frozen_v']


def independent_overlap(a,b):
    inter=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return inter/max(max(a[1],b[1])-min(a[0],b[0]),1e-12)


def independent_router(td,ids,ev):
    ps=ev['proposals'];cs=ev['proposal_confidence'];n=len(ps)
    sums=np.asarray([sum(independent_overlap(a,b) for j,b in enumerate(ps) if i!=j) for i,a in enumerate(ps)])
    agreement=sums/(n-1) if n>1 else np.zeros(n);q=np.asarray(cs)*agreement
    maps={k:np.zeros(len(ids)) for k in ['S','E','SE']}
    for f,fid in enumerate(ids):
        maps['S'][f]=sum(c['physical_interval'][0]<=fid<c['physical_interval'][1] for c in td['candidates'])/len(td['candidates'])
        maps['E'][f]=sum(v for v,(a,b) in zip(q,ps) if a<=fid<b)/q.sum() if q.sum()>0 else 0.
    maps['SE']=.5*(maps['S']+maps['E'])
    match=np.asarray([[independent_overlap(c['physical_interval'],p) for p in ps] for c in td['candidates']])
    scores={k:np.asarray([max([w*v for w,v in zip(weights,row)],default=0.) for row in match]) for k,weights in [('current',cs),('qc',q)]}
    return maps,agreement,q,sums,match,scores


def transitions(rows, before, after):
    out={}
    for metric in ['t','v']:
        out[metric]={}
        for th in [.3,.5]:
            a=np.asarray([r[before+'_'+metric]>th for r in rows]);b=np.asarray([r[after+'_'+metric]>th for r in rows])
            out[metric][str(th)]=dict(correct_before=int(a.sum()),wrong_before=int((~a).sum()),
                correct_to_wrong=int((a&~b).sum()),wrong_to_correct=int((~a&b).sum()),
                correct_unchanged=int((a&b).sum()),wrong_unchanged=int((~a&~b).sum()))
    return out


def support_summary(rows):
    out={}
    for condition in ['corruption','clean']:
        rr=group(rows,condition);out[condition]={}
        for router in ['S','E','SE']:
            fields=[router+'_'+k for k in SUPPORT_FIELDS]
            z=scalar_summary(rr,fields)
            z['counts']=dict(cells=len(rr),available=sum(r[router+'_statistics']['available'] for r in rr),
                missing_sampled_event=sum(r[router+'_statistics']['event_frames']==0 for r in rr),
                any_quantile_gt=sum(r[router+'_statistics']['quantile_any_gt'] for r in rr),
                quantile_duplicates=sum(r[router+'_statistics']['available'] and r[router+'_statistics']['quantile_unique']<5 for r in rr),
                nonempty_reference_arrivals=sum(r[router+'_statistics']['raw_valid_references']>0 for r in rr),
                zero_weighted_mass_nonempty=sum(r[router+'_statistics']['raw_valid_references']>0 and r[router+'_statistics']['valid_reference_mass']==0 for r in rr),
                useful_reference_discarded=sum(r[router+'_statistics']['useful_reference_discarded'] for r in rr))
            out[condition][router]=z
        for router in ['E','SE']:
            dr=[]
            for r in rr:
                x={k:r[k] for k in ['source_id','order','condition']}
                for k in SUPPORT_FIELDS:x['delta_'+k]=(r[router+'_'+k]-r['S_'+k]) if r[router+'_'+k] is not None and r['S_'+k] is not None else None
                dr.append(x)
            out[condition][router+'_minus_S']=scalar_summary(dr,['delta_'+k for k in SUPPORT_FIELDS])
    return out


def task_summary(rows, expert_rows):
    out={}
    for condition in ['corruption','clean']:
        out[condition]={}
        for subset in ['all','expert','nonexpert']:
            rr=group(rows,condition,subset);z=scalar_summary(rr,TASK_FIELDS)
            z['Current_to_QC']=transitions(rr,'Current','QC')
            delta=np.asarray([r['delta_v'] for r in rr]);z['cell_gross_gain_v_pp']=float(np.maximum(delta,0).mean()*100)
            z['cell_gross_loss_v_pp']=float(-np.minimum(delta,0).mean()*100)
            z['severe_harm_gt5pp']=int((delta<-.05).sum());z['severe_gain_gt5pp']=int((delta>.05).sum())
            if subset=='expert':
                z['Native_to_Current']=transitions(rr,'Native','Current');z['Native_to_QC']=transitions(rr,'Native','QC')
                er=group(expert_rows,condition)
                z['selection_regret']=scalar_summary(er,['Current_regret_t','Current_regret_v','QC_regret_t','QC_regret_v','delta_regret_t','delta_regret_v'])
                z['critic_diagnosis']={name:dict(bad_contributor=sum(r[name+'_bad_contributor'] for r in er),
                    selected_harms_native_t=sum(r[name+'_t']<r['Native_t']-1e-12 for r in er),
                    selected_harms_native_v=sum(r[name+'_v']<r['Native_v']-1e-12 for r in er),
                    missed_correct={m:{str(th):sum(max(r['candidate_'+m])>th and r[name+'_'+m]<=th for r in er) for th in [.3,.5]} for m in ['t','v']},
                    mean_contributor_regret=float(np.mean([r[name+'_contributor_regret'] for r in er if r[name+'_contributor_regret'] is not None]))) for name in ['Current','QC']}
            out[condition][subset]=z
    return out


def run():
    import torch
    torch.set_num_threads(2);t=time.time();lock=verify();bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['cells']==768 and bar['expert_cells']==192 and bar['GT_used'] is False
    for ds in DATASETS:
        assert sha(BASE/ds/'PREDICTION_BARRIER.json')==bar['datasets'][ds]
        assert sha(BASE/ds/'OBSERVATIONS.json')==read(BASE/ds/'PREDICTION_BARRIER.json')['file_sha256']
    from vg_tta.tastvg_temporal_router_t0_v1 import support_statistics
    from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
    from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
    from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
    from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
    from methods.decota_final_simplified_v1.tensors import state_hash
    results={};root_checks=collections.Counter();max_error=0.
    public=ROOT/'results/tastvg_temporal_router/2026-10-02'
    for ds in DATASETS:
        observations=read(BASE/ds/'OBSERVATIONS.json');p=read(PRIOR/ds/'PLAN.json')
        # This is the first new interpretation of diagnostic labels; maps and
        # both scoring-rule decisions have already been globally sealed.
        write(BASE/ds/'GT_EXPOSURE.json',dict(time=time.time(),barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
            GT_scope='exposed development offline diagnosis; no online selection or update',
            labels_sha256=sha(POOL/ds/'GT_LABELS_search.json'),globally_fresh=False))
        gt=read(POOL/ds/'GT_LABELS_search.json')
        old={ (r['condition'],r['order'],r['arrival']):r for r in read(PRIOR/ds/'A'/'ROWS.json') }
        source_ids={s:i for i,s in enumerate(sorted({p['rows'][i]['source'] for i in p['splits']['search']['orders']['order1']}))}
        rows=[];erows=[];previous={};boundary=0;metric=DenseMetric() if ds=='vidstg' else HC2DenseMetric()
        for obs in observations:
            x=load(ROOT/obs['payload']);row=p['rows'][obs['parent']];ids=np.asarray(row['frame_ids']);g=gt[str(obs['parent'])]
            truth={int(k):v for k,v in g['truth'].items()};span=g['span'];clip=ds=='hc2'
            key=obs['condition'],obs['order'];assert x['pre_sha']==obs['pre_sha']==state_hash(x['pre_state'])
            assert x['post_sha']==obs['post_sha']==state_hash(x['post_state'])
            expected=read(PRIOR/ds/'A'/'SUPPORT.json')['center_sha256'] if obs['arrival']==0 else previous[key]
            assert expected==x['pre_sha'];previous[key]=x['post_sha'];root_checks['state_links']+=1
            evaluate=evaluator(x['slow']['boxes'],row,truth,span,clip)
            fs=evaluator(x['source_native']['boxes'],row,truth,span,clip)(x['source_native']['indices'])
            native=evaluate(x['slow']['indices']);current=evaluate(obs['current_indices']);qc=evaluate(obs['qc_indices'])
            rr={k:obs[k] for k in ['parent','condition','order','arrival','expert_scheduled']};rr['source_id']=source_ids[row['source']]
            rr.update(pre_state_sha256=x['pre_sha'],post_state_sha256=x['post_sha'])
            for name,m in [('Frozen',fs),('Native',native),('Current',current),('QC',qc)]:rr.update({name+'_'+k:m[k] for k in ['t','v']})
            rr.update(delta_t=qc['t']-current['t'],delta_v=qc['v']-current['v'])
            for name,m in [('Current',current),('QC',qc)]:
                for k in ['t','v']:rr['delta_'+name+'_Frozen_'+k]=m[k]-fs[k]
                pix=xyxy(x['slow']['boxes'],row['input']['width'],row['input']['height']);pix=np.maximum(pix,0) if clip else pix
                ii=obs['current_indices'] if name=='Current' else obs['qc_indices'];interval=[ids[ii[0]],ids[ii[1]]+1]
                ref=metric(pix,ids,interval,truth,span);ind=dense_official_metrics(pix,ids,interval,truth,span)
                for k,code in [('m_tIoU','t'),('m_vIoU','v')]:
                    err=max(abs(ref[k]-ind[k]),abs(ref[k]-m[code]));assert err<1e-10;max_error=max(max_error,err);root_checks['dense_scalar_checks']+=2
            for code,k in [('t','m_tIoU'),('v','m_vIoU')]:
                assert abs(rr['Current_'+code]-old[(obs['condition'],obs['order'],obs['arrival'])]['Ours_'+k])<1e-10
                root_checks['prior_current_parity']+=1
            if obs['expert_scheduled']:
                ev={stage:load(POOL/ds/'experts'/read(ROOT/rel)['cache']) for stage,rel in obs['expert_receipts'].items()}
                td=x['temporal'];maps,a,q,sums,match,scores=independent_router(td,row['frame_ids'],ev['temporal']);router=obs['router']
                for name,array in [('proposal_agreement',a),('proposal_quality',q),('proposal_overlap_row_sums',sums),('candidate_proposal_overlap',match)]:
                    np.testing.assert_allclose(array,router[name],atol=1e-12,rtol=0);root_checks['proposal_numeric_values']+=array.size
                for name,w in maps.items():
                    np.testing.assert_allclose(w,router['weights'][name],atol=1e-12,rtol=0)
                    if w.sum()>0:
                        independent=[]
                        for quant in [.1,.3,.5,.7,.9]:
                            running=0.
                            for i,v in enumerate(w):
                                running+=v
                                if running>=quant*w.sum()-1e-14:independent.append(i);break
                        assert independent==router['quantiles'][name]
                    else:assert router['quantiles'][name] is None
                    root_checks['map_numeric_values']+=w.size
                for name,sc in scores.items():
                    np.testing.assert_allclose(sc,router[name+'_scores'],atol=1e-14,rtol=0)
                    assert int(np.argmax(sc))==router[name+'_selected'];root_checks['critic_values']+=len(sc)
                event=(ids>=span[0])&(ids<span[1]);scored=np.asarray([int(i) in truth for i in ids]);valid=np.asarray(ev['spatial']['valid'],bool)
                boundary+=int(not np.array_equal(event,scored));uniform=ev['spatial']['positions'];assert len(uniform)==5
                e={**rr,'candidate_count':len(td['candidates']),'proposal_count':len(q),
                    'duplicate_proposals':router['duplicate_proposals'],'proposal_confidence':ev['temporal']['proposal_confidence'],
                    'proposal_agreement':a.tolist(),'proposal_quality':q.tolist(),'proposal_overlap_row_sums':sums.tolist(),
                    'candidate_proposal_overlap':match.tolist(),'Current_scores':scores['current'].tolist(),'QC_scores':scores['qc'].tolist(),
                    'Current_selected':router['current_selected'],'QC_selected':router['qc_selected'],
                    'selection_changed':router['current_selected']!=router['qc_selected']}
                cm=[evaluate(c['indices']) for c in td['candidates']]
                for code in ['t','v']:
                    e['candidate_'+code]=[m[code] for m in cm]
                    for name in ['Current','QC']:e[name+'_regret_'+code]=max(e['candidate_'+code])-e[name+'_'+code]
                    e['delta_regret_'+code]=e['QC_regret_'+code]-e['Current_regret_'+code]
                prop_gt=np.asarray([independent_overlap(j,span) for j in ev['temporal']['proposals']]);e['proposal_gt_t']=prop_gt.tolist()
                for name,weights in [('Current',np.asarray(ev['temporal']['proposal_confidence'])),('QC',q)]:
                    selected=e[name+'_selected'];contribution=match[selected]*weights
                    wi=int(np.argmax(contribution)) if len(contribution) and max(contribution)>0 else None
                    e[name+'_winning_contributor']=wi;e[name+'_bad_contributor']=bool(wi is not None and prop_gt[wi]<=.3 and prop_gt.max()>.5)
                    e[name+'_contributor_regret']=float(prop_gt.max()-prop_gt[wi]) if wi is not None else None
                for name,w in maps.items():
                    st=support_statistics(w,event,scored,valid,router['quantiles'][name],uniform)
                    e[name+'_statistics']=st;e.update({name+'_'+k:st[k] for k in SUPPORT_FIELDS})
                    # Independent sums/denominators are checked from raw masks.
                    assert abs(st['event_mass']-sum(v for v,mask in zip(w,event) if mask))<1e-12
                    assert abs(st['valid_reference_mass']-sum(v for v,mask in zip(w,valid) if mask))<1e-12
                    root_checks['support_statistics']+=1
                erows.append(e)
            else:assert obs['qc_indices']==obs['current_indices'] and rr['delta_v']==rr['delta_t']==0.
            rows.append(rr)
        assert len(rows)==384 and len(erows)==96
        ss=support_summary(erows);ts=task_summary(rows,erows)
        checks={router:dict(mass_gain=ss['corruption'][router+'_minus_S']['metrics']['delta_gt_mass']['mean'],
                    reference_gain=ss['corruption'][router+'_minus_S']['metrics']['delta_quantile_coverage']['mean'],
                    recall_gain=ss['corruption'][router+'_minus_S']['metrics']['delta_event_recall']['mean']) for router in ['E','SE']}
        ct=ts['corruption']['expert'];protected=all(ct['Native_to_QC'][m][th]['correct_to_wrong']<=ct['Native_to_Current'][m][th]['correct_to_wrong'] for m in ['t','v'] for th in ['0.3','0.5'])
        signal=any(v['mass_gain']>1e-12 and v['reference_gain']>1e-12 and v['recall_gain']>=-1e-12 for v in checks.values()) and ct['metrics']['delta_t']['mean']>1e-12 and protected
        decision=dict(T1_development_signal=signal,support_checks=checks,QC_mean_t_gain=ct['metrics']['delta_t']['mean'],native_correct_destruction_not_increased=protected,
            H_full_started=False,new_model_execution=False,spatial_A_changed=False)
        config=dict(**read(BASE/'INVENTORY.json')[ds],
            GT_data='historically exposed development labels, read after global map/decision seal',
            current_source_commit='5d6da2bb6d800e0766f548752a58761e7ed244c9',
            sampled_event_vs_scored_frame_mask_different=boundary,
            aggregation='source/order/condition macro; paired10000 bootstrap seed20261001',
            task_type='CPU support audit and exact readout of fixed spatial A online trajectory')
        results[ds]=dict(support=ss,task=ts,decision=decision)
        for name,obj in [('CONFIG',config),('ROWS',rows),('EXPERT_ROWS',erows),('SUPPORT_SUMMARY',ss),('TASK_SUMMARY',ts),('DECISION',decision)]:
            write(BASE/ds/(name+'.json'),obj);write(public/ds/(name+'.json'),obj)
        print('SCORED',ds,'QC expert delta_t/v pp',ct['metrics']['delta_t']['mean']*100,ct['metrics']['delta_v']['mean']*100,'T1 signal',signal,flush=True)
    assert not torch.cuda.is_initialized()
    gate=any(z['decision']['T1_development_signal'] for z in results.values())
    conclusion=dict(status='T1_exact_CPU_readout_released' if gate else 'T0_completed_T1_not_triggered',
        T1_triggered=gate,new_GPU_trial=False,H_full_started=False,production_promoted=False,
        signal_datasets=[ds for ds in DATASETS if results[ds]['decision']['T1_development_signal']])
    audit=dict(status='pass',checks=dict(root_checks),max_dense_error=max_error,GT_after_global_seal=True,
        source_current_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        new_model_forward_calls=0,new_backward_calls=0,new_expert_calls=0,GPU_initialized=False,
        CPU_scoring_wall_seconds=time.time()-t,observations=768,expert_maps=192,
        pipeline_equivalence='QC only changes final indices; original spatial targets/update/state/ordering remain exact sealed A')
    write(BASE/'ROOT_READBACK.json',audit);write(public/'ROOT_READBACK.json',audit)
    write(BASE/'DECISION.json',conclusion);write(public/'DECISION.json',conclusion)
    write(public/'RESOURCES.json',dict(new_GPU_calls=0,new_expert_calls=0,new_forward_backward=0,
        map_CPU_wall_seconds=bar['seconds'],score_CPU_wall_seconds=time.time()-t,
        old_sealed_outputs_reused=768,new_expert_support_maps=192))
    status(BASE/'STATUS.json',dict(status='completed_pending_public_audit_report_GitHub',done=768,total=768,GT_used=True))


if __name__=='__main__':run()
