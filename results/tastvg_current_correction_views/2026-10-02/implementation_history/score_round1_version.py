"""Post-stage-barrier CPU root audit, paired metrics and decision sealing."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_correction_views_common_v1 import *

def aggregate(rows,arms):
    from scripts.score_tastvg_best_quick_v1 import source_summary
    result={}
    for group in ['corruption','clean']:
        result[group]={}
        for sub in ['all','expert','nonexpert']:
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and
                (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
            fields=['Frozen_v','A_v']+[a+'_'+m for a in arms for m in ['v','t','s']]+['delta_'+a+'_v' for a in arms]
            z=source_summary(rr,fields);z['effects']={}
            for a in arms:
                key='delta_'+a+'_v';take=[dict(r,gross_gain=max(r[key],0.),gross_loss=max(-r[key],0.)) for r in rr]
                z['effects'][a]=dict(positive=sum(r[key]>1e-12 for r in rr),negative=sum(r[key]<-1e-12 for r in rr),
                    zero=sum(abs(r[key])<=1e-12 for r in rr),harm_gt5pp=sum(r[key]<-.05 for r in rr),
                    gross=source_summary(take,['gross_gain','gross_loss']),
                    correctness={str(t):dict(rescued=sum(r['A_v']<=t<r[a+'_v'] for r in rr),
                        destroyed=sum(r[a+'_v']<=t<r['A_v'] for r in rr)) for t in [.3,.5]})
            result[group][sub]=z
    return result

def pooled(allrows,field):
    matrices=[]
    for ds in DATASETS:
        rr=[r for r in allrows[ds] if r['condition']!='clean'];sources=sorted({r['source_id'] for r in rr})
        matrices.append(np.array([np.mean([r[field] for r in rr if r['source_id']==s]) for s in sources]))
    rng=np.random.default_rng(20261001);boots=[]
    for i in range(100):
        boots.extend(.5*sum(a[rng.integers(0,len(a),(100,len(a)))].mean(1) for a in matrices))
    return dict(mean=float(.5*sum(a.mean() for a in matrices)),ci95=np.quantile(boots,[.025,.975]).tolist(),
        dataset_means={d:float(a.mean()) for d,a in zip(DATASETS,matrices)},equal_dataset_weight=True)

def run(stage,split):
    import torch
    torch.set_num_threads(2)
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.audit_tastvg_ur_write_v1 import independent_rewards,independent_geometry,logsoftmax,ranks
    from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
    from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
    from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
    from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
    from vg_tta.tastvg_reference_selection_v1 import student_frames,pairwise
    from vg_tta.tastvg_current_correction_views_v1 import select,temporal
    verify();gb=BASE/f'{stage}_{split}_GLOBAL_PREDICTION_BARRIER.json';globalbar=read(gb)
    sl=read(BASE/'SCORING_RUNTIME_LOCK.json');sp=dict(sl['pins'])
    for rf in sorted((BASE/'scoring_revisions').glob('*.json')):sp.update(read(rf)['pin_overrides'])
    for f,h in sp.items():assert sha(ROOT/f)==h
    assert globalbar['status']=='sealed' and globalbar['GT_read'] is False
    if split=='confirm':
        assert (BASE/'round1_confirm_GLOBAL_PREDICTION_BARRIER.json').exists() and (BASE/'round2_confirm_GLOBAL_PREDICTION_BARRIER.json').exists()
        assert read(BASE/'FINAL_SELECTION.json')['time']<globalbar['time']
    directory=PUBLIC/stage/split;allrows={};checks=collections.Counter();maxerr=0.;allwrites={}
    c=read(BASE/'ROUND1_SELECTION.json')['correction'] if stage=='round2' else None
    rbranch={'R_select':'Rnew_select','R_temp':'Rnew_temp','Specific_temp':'Rnew_specific'}.get(c,c)
    arms=ARMS if stage=='round1' else ['A','C','T','CT','C_newacquisition','CT_newacquisition','twoUniform5','R_acquisition_old','R_acquisition_new']
    for ds in DATASETS:
        p=plan(ds);cfg=p['params'];out=BASE/ds/stage/split;bar=read(out/'PREDICTION_BARRIER.json')
        assert sha(out/'PREDICTION_BARRIER.json')==globalbar['datasets'][ds] and bar['time']<=globalbar['time']
        for f,h in bar['files'].items():assert sha(out/f)==h and sha((out/f).with_suffix('.pt'))==read(out/f)['sha256']
        # Labels are opened only after both datasets have sealed this stage.
        labels=read(POOL/ds/f'GT_LABELS_{split}.json')
        write(BASE/ds/stage/split/'GT_EXPOSURE.json',dict(scope='post-stage sealed CPU metric and diagnosis',
            barrier_sha256=sha(gb),historically_exposed=True,confirmation_not_selected=split=='confirm',time=time.time()))
        metric=DenseMetric() if ds=='vidstg' else HC2DenseMetric();rows=[];writes=[]
        def score(pred,parent):
            nonlocal maxerr
            row=p['rows'][parent];g=labels[str(parent)];truth={int(k):v for k,v in g['truth'].items()};ids=row['frame_ids'];span=g['span']
            b=xyxy(pred['boxes'],row['input']['width'],row['input']['height']);b=np.maximum(b,0) if ds=='hc2' else b
            idx=pred['indices'];interval=[ids[idx[0]],ids[idx[1]]+1]
            if 'physical_interval' in pred:assert pred['physical_interval']==interval
            m=metric(b,ids,interval,truth,span);z=dense_official_metrics(b,ids,interval,truth,span)
            for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5']:
                err=abs(m[k]-z[k]);assert err<1e-10;maxerr=max(maxerr,err);checks['independent_metric_scalars']+=1
            e=evaluator(pred['boxes'],row,truth,span,ds=='hc2')(idx)
            assert abs(e['v']-m['m_vIoU'])<1e-10
            return dict(v=float(m['m_vIoU']),t=float(m['m_tIoU']),s=float(m['sIoU_dense_GT']))
        for cond in p['conditions']:
            for order,seq in p['splits'][split]['orders'].items():
                previous=None
                for at,parent in enumerate(seq):
                    x=checked(out/'predictions'/cond/order/f'{at:05}.pt');cell=x['cell'];old=oldcell(ds,split,cond,order,at)
                    assert cell['parent']==parent and x['persistent_unchanged'] and x['GT_read'] is False
                    assert x['persistent_pre_sha']==old['pre_sha'] and x['persistent_post_sha']==old['post_sha']
                    assert previous is None or previous==old['pre_sha'];previous=old['post_sha'];checks['persistent_links']+=1
                    assert x['persistent_payload_sha256']==sha(ROOT/cell['old_payload'])
                    outputs=x['predictions'];scheduled=cell['scheduled'];ids=p['rows'][parent]['frame_ids']
                    if stage=='round2':
                        a=checked(BASE/ds/'round1'/split/'predictions'/cond/order/f'{at:05}.pt');oldc=a['predictions'][c]
                        if scheduled:
                            prefix=f'{split}_{cond}_{order}_{at:05}';view=read(BASE/ds/'views'/f'{prefix}.json')
                            e0,_=expert(ds,'temporal',parent,cond,old['pixel_sha256']);assert sha(ROOT/view['cache'])==view['cache_sha256']
                            e1=load(ROOT/view['cache']);tr=temporal(old['temporal']['candidates'],[e0,e1]);assert tr==view['rule'];checks['temporal_comparisons']+=1
                            # Independent scalar interval IoU, not online critic code.
                            sc=[]
                            for ev in [e0,e1]:
                                line=[]
                                for cand in old['temporal']['candidates']:
                                    aa,bb=cand['physical_interval'];terms=[]
                                    for (cc,dd),weight in zip(ev['proposals'],ev['proposal_confidence']):
                                        inter=max(0.,min(bb,dd)-max(aa,cc));union=bb-aa+dd-cc-inter
                                        terms.append(weight*inter/max(union,1e-12))
                                    line.append(max(terms,default=0.))
                                sc.append(line)
                            sc=np.asarray(sc);dv=sc-sc[:,:1];worst=dv.min(0);top=np.flatnonzero(worst==worst.max())
                            selected=int(top[0]) if len(top)==1 and worst[top[0]]>0 else 0
                            np.testing.assert_allclose(sc,tr['scores'],atol=1e-12,rtol=0);assert selected==tr['selected']
                            mapped=np.asarray(e1['seconds_segments'])*e1['fps']+e1['physical_origin']
                            np.testing.assert_allclose(mapped,e1['proposals'],atol=1e-5,rtol=1e-6)
                            checks['independent_temporal_scalars']+=sc.size+mapped.size
                            idx=old['temporal']['candidates'][tr['selected']]['indices'];newc=outputs[rbranch] if c!='A' else outputs['A']
                            u2name={'R_select':'U2_select','R_temp':'U2_temp','Specific_temp':'U2_specific','U_select':'U2_select','A':'U2_select'}[c]
                            def retime(z,newidx):return dict(z,indices=newidx,physical_interval=[ids[newidx[0]],ids[newidx[1]]+1])
                            outputs=dict(A=outputs['A'],C=oldc,T=retime(outputs['A'],idx),CT=retime(oldc,idx),
                                C_newacquisition=newc,CT_newacquisition=retime(newc,idx),twoUniform5=x['predictions'][u2name],
                                R_acquisition_old=a['predictions']['R_select'],R_acquisition_new=x['predictions']['Rnew_select'])
                        else:outputs={name:x['A'] for name in arms}
                    record=dict(parent=parent,source_id=parent,dataset=ds,condition=cond,order=order,arrival=at,expert_scheduled=scheduled)
                    frozen=score(old['source_native'],parent);record.update(Frozen_v=frozen['v'],Frozen_t=frozen['t'])
                    for arm in arms:
                        s=score(outputs[arm],parent);record.update({arm+'_'+k:v for k,v in s.items()})
                    for arm in arms:record['delta_'+arm+'_v']=record[arm+'_v']-record['A_v']
                    if stage=='round2':
                        record['delta_acquisition_v']=record['C_newacquisition_v']-record['C_v']
                        record['delta_R_acquisition_v']=record['R_acquisition_new_v']-record['R_acquisition_old_v']
                        record['delta_temporal_t']=record['T_t']-record['A_t']
                    rows.append(record)
                    if not scheduled:
                        assert all(abs(record['delta_'+arm+'_v'])<1e-12 for arm in arms);checks['nonexpert_identity']+=1;continue
                    assert state_hash(x['pre_state'])==old['pre_sha'];checks['current_state_bindings']+=1
                    probev=[score(dict(z,indices=x['A']['indices'],physical_interval=x['A']['physical_interval']),parent)['v'] for z in x['probes']]
                    wr={k:record[k] for k in ['parent','source_id','condition','order','arrival']};wr['probe_v']=probev;wr['evidence']={}
                    for branch,r in x['evidence'].items():
                        assert sha(ROOT/r['cache'])==r['cache_sha256'];e=load(ROOT/r['cache']);reward=independent_rewards(x['probes'],e)
                        assert select(reward)==x['selections'][branch]
                        if reward is None:assert x['rewards'][branch] is None
                        else:
                            np.testing.assert_allclose(reward,x['rewards'][branch],atol=1e-12,rtol=0);checks['reward_scalars']+=9
                            d=independent_geometry(x['A']['boxes'],[z['boxes'] for z in x['probes']],x['coefficients'])
                            lp=logsoftmax(-d/cfg['student_temperature']);lq=logsoftmax(-ranks(reward)/cfg['teacher_temperature']);m=x['metadata'][branch]
                            np.testing.assert_allclose(d,m['distances'],atol=2e-5,rtol=2e-5)
                            np.testing.assert_allclose(np.exp(lp),m['p'],atol=2e-6,rtol=2e-5)
                            np.testing.assert_allclose(np.exp(lq),m['q'],atol=2e-7,rtol=1e-6)
                            np.testing.assert_allclose(np.sum(np.exp(lp)*(lp-lq)),m['loss'],atol=2e-5,rtol=2e-5);checks['KL_checks']+=28
                        positions=e['positions'];g=labels[str(parent)];lo,hi=g['span'];prec=np.mean([lo<=ids[i]<hi for i in positions])
                        valid=np.asarray(e['valid'],bool);valid_event=sum(valid[i] and lo<=ids[i]<hi for i in positions)
                        wr['evidence'][branch]=dict(rewards=reward,selection=select(reward),event_frame_precision=float(prec),
                            valid_frames=int(valid.sum()),valid_event_frames=int(valid_event),positions=positions,
                            critic_pairwise=pairwise(reward,probev),selected_v=probev[select(reward)['selected']],
                            new_call=r.get('new_call',False),seconds=r.get('seconds',0.),input_sha256=r['input_sha256'])
                    for name,st in x['temporary_states'].items():
                        branch=name.split('_')[0];ok=x['rewards'][branch] is not None
                        if name.endswith('specific'):
                            ok=ok and x['rewards']['U'] is not None;grad={n:x['gradients'][branch][n]-x['gradients']['U'][n] for n in st}
                        else:grad=x['gradients'][branch]
                        for n,v in x['pre_state'].items():
                            expected=(v.double()-cfg['lr']*grad[n].double()).float() if ok else v
                            torch.testing.assert_close(expected,st[n],rtol=2e-6,atol=1e-6);checks['SGD_coordinates']+=v.numel()
                    if stage=='round1':
                        expected=student_frames(old['temporal']['candidates'],ids)['positions'];assert wr['evidence']['R']['positions']==expected
                        for branch in ['U','R']:
                            arm=branch+'_select';j=x['selections'][branch]['selected'];assert torch.equal(x['predictions'][arm]['boxes'],x['probes'][j]['boxes'])
                        wr['execution']=dict(R_selected_better_than_A=record['R_select_v']>record['A_v']+1e-12,
                            R_temp_harm=record['R_temp_v']<record['A_v']-1e-12,
                            Specific_temp_harm=record['Specific_temp_v']<record['A_v']-1e-12,
                            R_selected_good_update_harm=record['R_select_v']>record['A_v']+1e-12 and record['R_temp_v']<record['A_v']-1e-12)
                    else:
                        expected=student_frames([old['temporal']['candidates'][tr['selected']]],ids)['positions'];assert wr['evidence']['Rnew']['positions']==expected
                        wr['temporal']=dict(rule=tr,actual_observation_difference=view['actual_observation_difference'],
                            original_selected=old['temporal']['selected'],new_selected=tr['selected'])
                    writes.append(wr)
        allrows[ds]=rows;allwrites[ds]=writes
        write(directory/ds/'ROWS.json',rows);write(directory/ds/'WRITE_ROWS.json',writes)
        write(directory/ds/'SUMMARY.json',aggregate(rows,arms))
        write(directory/ds/'CASES.json',{a:dict(positive=sorted([r for r in rows if r['condition']!='clean'],key=lambda r:-r['delta_'+a+'_v'])[:5],
            negative=sorted([r for r in rows if r['condition']!='clean'],key=lambda r:r['delta_'+a+'_v'])[:5]) for a in arms})
    effects={a:pooled(allrows,'delta_'+a+'_v') for a in arms};write(directory/'POOLED.json',effects)
    audit=dict(status='pass',stage=stage,split=split,checks=dict(checks),max_metric_error=maxerr,
        after_global_prediction_barrier=True,barrier_sha256=sha(gb),CUDA_initialized=torch.cuda.is_initialized(),time=time.time())
    assert not audit['CUDA_initialized'];write(BASE/f'{stage}_{split}_ROOT_AUDIT.json',audit);write(directory/'ROOT_AUDIT.json',audit)
    if split=='search':
        if stage=='round1':
            eligible=[a for a in ARMS if effects[a]['mean']>0 and min(effects[a]['dataset_means'].values())>=0]
            chosen=max(eligible,key=lambda a:effects[a]['mean']) if eligible else 'A'
            selection=dict(correction=chosen,eligible=eligible,effects=effects,rule='single pooled correction, nonnegative both datasets',time=time.time())
            write(BASE/'ROUND1_SELECTION.json',selection);write(PUBLIC/'ROUND1_SELECTION.json',selection)
        else:
            ae=pooled(allrows,'delta_acquisition_v');te=pooled(allrows,'delta_temporal_t')
            eligible_acq=c.startswith('R') or c=='Specific_temp'
            use_acq=eligible_acq and ae['mean']>0 and min(ae['dataset_means'].values())>=0
            candidates=['A','C','T','CT']+(['C_newacquisition','CT_newacquisition'] if use_acq else [])
            eligible=[a for a in candidates if effects[a]['mean']>0 and min(effects[a]['dataset_means'].values())>=0]
            chosen=max(eligible,key=lambda a:effects[a]['mean']) if eligible else 'A'
            selection=dict(final=chosen,correction=c,new_acquisition_eligible=bool(use_acq),acquisition_effect=ae,
                temporal_t_effect=te,candidates=candidates,effects=effects,rule='single rule for both datasets, confirmed without reselection',time=time.time())
            write(BASE/'FINAL_SELECTION.json',selection);write(PUBLIC/'FINAL_SELECTION.json',selection)
    print('AUDITED',stage,split,'cells',sum(map(len,allrows.values())),'max metric error',maxerr,flush=True)
    status(BASE/'STATUS.json',dict(status='scored_pending_next_stage_or_publication',stage=stage,split=split,time=time.time()))
if __name__=='__main__':run(*sys.argv[1:])
