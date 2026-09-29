"""Sealed four-arm task scores and independent J0 numerical/call-schedule audit."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_schedule_j01_v1 import OUT,J0,S0,TEMP,NATIVE,verify
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.score_tastvg_spatial_online_opd_s1_v1 import macro
from methods.decota_final_simplified_v1.tensors import state_hash
from scripts.score_tastvg_spatial_online_opd_s1_v1 import write_exposure
ARMS={'Frozen':'frozen','Fast-only':'fast','Slow-only':'slow','Final':'final'}
PAIRS=[('fast','frozen'),('slow','frozen'),('final','frozen'),('final','fast'),('final','slow')]


def independent_scores(candidates,teacher):
    values=[]
    for c in candidates:
        a,b=c['physical_interval'];scores=[]
        for (start,end),conf in zip(teacher['proposals'],teacher['proposal_confidence']):
            overlap=max(0.,min(b,end)-max(a,start));union=max(b,end)-min(a,start);scores.append(overlap/max(union,1e-12)*conf)
        values.append(max(scores) if scores else 0.)
    return np.array(values)


def run():
    import ijson
    torch.set_num_threads(4);p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==480
    for f,h in {**bar['files'],**bar['final_states']}.items():assert sha(OUT/f)==h
    keys={r['key'] for r in p['rows']};lp=ROOT/'artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json'
    with lp.open('rb') as f:gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
    assert set(gt)==keys;write_exposure(OUT/'GT_EXPOSURE.json',dict(time=time.time(),queries=16,old_exposed_only=True,prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),GT_container_sha256=sha(lp)))
    initial=load(NATIVE/'PARAMETER_SUPPORT.pt')['center'];rows=[];metriccalls=paramchecks=klchecks=teacherchecks=0;maxparam=maxkl=0.;changed_support=changed_interval=gradchecks=0;expert_sets=empty=nonexpert=0;postdiff=0
    for order,cond in [(o,c) for o in p['orders'] for c in p['conditions']]:
        previous=initial
        for arrival,parent in enumerate(p['orders'][order]):
            r=next(v for v in p['rows'] if v['ordinal']==parent)
            name=f"{r['ordinal']:03}.pt";x=load(OUT/'online'/order/cond/name);assert x['expert_scheduled']==(arrival in p['expert_indices'])
            assert state_hash(x['pre_state'])==x['pre_state_sha256'] and state_hash(x['post_state'])==x['post_state_sha256'];assert all(torch.equal(v,previous[n]) for n,v in x['pre_state'].items());previous=x['post_state']
            assert torch.equal(x['arms']['Final']['boxes'],x['prediction']['boxes']) and torch.equal(x['arms']['Final']['boxes'],x['arms']['Slow-only']['boxes'])
            if x['expert_scheduled']:
                assert x['expert_reads']==['temporal','spatial'];expert_sets+=1;changed_support+=x['temporal_support_changed'];changed_interval+=x['temporal_selected_interval_changed']
                teacher=load(TEMP/'c2'/cond/name);assert teacher['pixel_sha256']==x['pixel_sha256']
                for field,arm in [('temporal','Final'),('fast_control','Fast-only')]:
                    d=x[field];scores=independent_scores(d['candidates'],teacher);np.testing.assert_allclose(scores,d['scores'],atol=1e-14,rtol=0);selected=int(np.argmax(scores));assert selected==d['selected'];assert x['arms'][arm]['indices']==d['candidates'][selected]['indices'];assert not d['teacher_interval_as_output'];teacherchecks+=len(scores)
                if x['rewards'] is None:empty+=1;assert not x['updated']
            else:
                nonexpert+=1;assert x['expert_reads']==[] and 'temporal' not in x and 'candidates' not in x
                assert x['arms']['Final']['indices']==x['arms']['Slow-only']['indices'];assert x['arms']['Fast-only']['indices']==x['arms']['Frozen']['indices']
            if x['updated']:
                u=x['update'];d=np.array(u['distances'].double());pp=np.exp(-d+d.min());pp/=pp.sum();re=np.array(x['rewards']);ranked_indices=np.argsort(-re);rank=np.empty(9);i=0
                while i<9:
                    j=i+1
                    while j<9 and re[ranked_indices[i]]-re[ranked_indices[j]]<=1e-12:j+=1
                    rank[ranked_indices[i:j]]=(i+j-1)/2;i=j
                qq=np.exp(-rank);qq/=qq.sum();kl=float(np.sum(pp*np.log(pp/qq)));err=abs(kl-u['loss_before']);assert err<1e-6;maxkl=max(maxkl,err);klchecks+=1
                np.testing.assert_allclose(pp,u['p'],atol=1e-7,rtol=0);np.testing.assert_allclose(qq,u['q'],atol=1e-7,rtol=0)
                for n,b in x['pre_state'].items():
                    g=u['gradients'][n];expected=(b.double().numpy()-.005*g.double().numpy()).astype(np.float32);err=float(np.max(np.abs(expected-x['post_state'][n].numpy())));assert err<=1.5e-7;maxparam=max(maxparam,err);paramchecks+=g.numel()
                postdiff+=int(not torch.equal(x['post_prediction']['boxes'],x['arms']['Final']['boxes']))
            else:assert all(torch.equal(v,x['pre_state'][n]) for n,v in x['post_state'].items())
            rec=dict(order=order,parent=r['ordinal'],condition=cond,arrival=arrival,expert_scheduled=x['expert_scheduled'],updated=x['updated'],pre_state_sha256=x['pre_state_sha256'],post_state_sha256=x['post_state_sha256'],step_norm=x['parameter_displacement'],inherited_state_norm=x['displacement_from_source'],temporal_support_changed=x.get('temporal_support_changed',False),temporal_selected_interval_changed=x.get('temporal_selected_interval_changed',False),current_spatial_pre_update=True)
            if x['updated']:rec['update_diagnostics']={k:(v.tolist() if torch.is_tensor(v) else v) for k,v in x['update'].items() if k!='gradients'}
            if x['expert_scheduled']:
                # No physical time/interval coordinates or expert proposal tensors in public scalar table.
                rec['temporal_diagnostics']={field:dict(scores=x[field]['scores'],selected=x[field]['selected'],candidate_count=len(x[field]['candidates'])) for field in ['temporal','fast_control']}
            for arm,short in ARMS.items():
                pred=x['arms'][arm];m,_=checked_score(pred['boxes'],gt[r['key']],r['frame_ids'],pred['indices']);metriccalls+=1
                for code,key in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:rec[f'{short}_{code}']=m[key]
            for short,m in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:
                for a,b in PAIRS:rec[f'{a}_minus_{b}_{short}']=rec[a+'_'+short]-rec[b+'_'+short]
                rec['interaction_'+short]=rec['final_'+short]-rec['fast_'+short]-rec['slow_'+short]+rec['frozen_'+short]
                if not x['expert_scheduled']:assert abs(rec['final_minus_fast_'+short]-rec['slow_minus_frozen_'+short])<1e-12
            rows.append(rec)
        assert all(torch.equal(v,previous[n]) for n,v in load(OUT/'final_states'/order/f'{cond}.pt').items())
    summary={};effects=[];keys=[k for k in rows[0] if k.endswith(('_s','_t','_v'))]
    for order in p['orders']:
        summary[order]={}
        for group in ['corruption']+p['conditions']:
            rr=[r for r in rows if r['order']==order and (r['condition']!='clean' if group=='corruption' else r['condition']==group)];summary[order][group]={}
            for subset in ['all','expert','nonexpert']:
                seq=[r for r in rr if subset=='all' or r['expert_scheduled']==(subset=='expert')]
                summary[order][group][subset]=dict(cells=len(seq),sources=len({r['parent'] for r in seq}),metrics={k:macro(seq,k) for k in keys})
                if group in ['corruption','clean']:
                    for parent in sorted({r['parent'] for r in seq}):effects.append(dict(order=order,parent=parent,group=group,subset=subset,**{k:float(np.mean([r[k] for r in seq if r['parent']==parent])) for k in keys if '_minus_' in k or 'interaction_' in k}))
    reference=read(J0/'SUMMARY.json');across={}
    def describe(values):
        a=np.array(values,float);return dict(n_orders=len(a),values=a.tolist(),mean=float(a.mean()),sample_std=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()),positive=int((a>0).sum()),negative=int((a<0).sum()),zero=int((a==0).sum()))
    for group in ['corruption']+p['conditions']:
        across[group]={}
        for subset in ['all','expert','nonexpert']:
            across[group][subset]={k:describe([summary[o][group][subset]['metrics'][k]['mean'] for o in p['orders']]) for k in keys}
    including={g:{sub:{k:describe([summary[o][g][sub]['metrics'][k]['mean'] for o in p['orders']]+[reference[g][sub]['metrics'][k]['mean']]) for k in keys} for sub in ['all','expert','nonexpert']} for g in ['corruption','clean']}
    write(OUT/'ROWS.json',rows);write(OUT/'SUMMARY.json',summary);write(OUT/'SOURCE_EFFECTS.json',effects);write(OUT/'ACROSS_ORDERS.json',across);write(OUT/'INCLUDING_J0.json',including);write(OUT/'J0_REFERENCE.json',{g:reference[g] for g in ['corruption','clean']})
    write(OUT/'AUDIT.json',dict(status='pass',state_links=480,source_resets=30,independent_SGD_coordinates=paramchecks,maximum_parameter_error=maxparam,independent_KL_updates=klchecks,maximum_KL_error=maxkl,KL_tolerance=1e-6,parameter_tolerance=1.5e-7,dual_metric_calls=metriccalls,independent_temporal_scores=teacherchecks,scheduled_expert_cells=expert_sets,nonexpert_no_evidence_reads=nonexpert,empty_spatial_noops=empty,current_output_pre_update_verified=480,updated_post_boxes_differ_from_reported_pre_boxes=postdiff,current_temporal_support_changed_cells=changed_support,current_temporal_selected_interval_changed_cells=changed_interval,spatial_full_reinsertions=4,temporal_all_layer_reinsertions=2,independent_Slow_replay=read(OUT/'SLOW_REPLAY_AUDIT.json'),all_predictions_before_GT=True,method_hashes_unchanged=True))
    allocations=[read(f) for f in (OUT/'allocations').glob('*.json')];replay=read(OUT/'SLOW_REPLAY_AUDIT.json')
    write(OUT/'RESOURCES.json',dict(new_GPU_process_seconds=sum(x['seconds'] for x in allocations),GPU_failed_attempts=sum(x['status']!='completed' for x in allocations),main_backward_steps=klchecks,validation_backward_steps=replay['updates'],main_spatial_candidates=9*expert_sets,validation_spatial_candidates=36,new_encoder_captures=0,new_expert_inferences=0,new_Final_arrivals=480,independent_Slow_validation_arrivals=16,logical_specialist_calls_per_order={'Frozen':{'temporal':0,'spatial':0},'Fast-only':{'temporal':24,'spatial':0},'Slow-only':{'temporal':0,'spatial':24},'Final':{'temporal':24,'spatial':24}},availability_fraction=.25))
    for order in p['orders']:print(order,{sub:{k:100*summary[order]['corruption'][sub]['metrics'][k]['mean'] for k in ['fast_minus_frozen_v','final_minus_fast_s','final_minus_fast_v','final_minus_frozen_v']} for sub in ['all','nonexpert']})
    print('ACROSS',{sub:{k:across['corruption'][sub][k] for k in ['fast_minus_frozen_v','final_minus_fast_s','final_minus_fast_v','final_minus_frozen_v']} for sub in ['all','nonexpert']})

if __name__=='__main__':run()
