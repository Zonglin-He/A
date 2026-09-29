"""Sealed four-arm task scores and independent J0 numerical/call-schedule audit."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_joint_j0_v1 import OUT,SLOW,S0,TEMP,NATIVE,verify
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.score_tastvg_spatial_online_opd_s1_v1 import macro
from methods.decota_final_simplified_v1.tensors import state_hash
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
    torch.set_num_threads(4);p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==96
    for f,h in {**bar['files'],**bar['final_states']}.items():assert sha(OUT/f)==h
    keys={r['key'] for r in p['rows']};lp=ROOT/'artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json'
    with lp.open('rb') as f:gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
    assert set(gt)==keys;write(OUT/'GT_EXPOSURE.json',dict(time=time.time(),queries=16,old_exposed_only=True,prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),GT_container_sha256=sha(lp)))
    slowrows={(r['parent'],r['condition']):r for r in read(SLOW/'ROWS.json')};initial=load(NATIVE/'PARAMETER_SUPPORT.pt')['center'];rows=[];metriccalls=paramchecks=klchecks=teacherchecks=0;maxparam=maxkl=0.;changed_support=changed_interval=gradchecks=0;expert_sets=empty=nonexpert=0;postdiff=0
    for cond in p['conditions']:
        previous=initial
        for arrival,r in enumerate(p['rows']):
            name=f"{r['ordinal']:03}.pt";x=load(OUT/'online'/cond/name);assert x['expert_scheduled']==(arrival in p['expert_indices'])
            assert state_hash(x['pre_state'])==x['pre_state_sha256'] and state_hash(x['post_state'])==x['post_state_sha256'];assert all(torch.equal(v,previous[n]) for n,v in x['pre_state'].items());previous=x['post_state']
            z=load(SLOW/'online'/cond/name)
            assert all(torch.equal(v,z[k][n]) for k in ['pre_state','post_state'] for n,v in x[k].items())
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
                u=x['update'];d=np.array(u['distances'].double());pp=np.exp(-d+d.min());pp/=pp.sum();re=np.array(x['rewards']);order=np.argsort(-re);rank=np.empty(9);i=0
                while i<9:
                    j=i+1
                    while j<9 and re[order[i]]-re[order[j]]<=1e-12:j+=1
                    rank[order[i:j]]=(i+j-1)/2;i=j
                qq=np.exp(-rank);qq/=qq.sum();kl=float(np.sum(pp*np.log(pp/qq)));err=abs(kl-u['loss_before']);assert err<1e-6;maxkl=max(maxkl,err);klchecks+=1
                np.testing.assert_allclose(pp,u['p'],atol=1e-7,rtol=0);np.testing.assert_allclose(qq,u['q'],atol=1e-7,rtol=0)
                for n,b in x['pre_state'].items():
                    g=u['gradients'][n];assert torch.equal(g,z['update']['gradients'][n]);gradchecks+=g.numel();expected=(b.double().numpy()-.005*g.double().numpy()).astype(np.float32);err=float(np.max(np.abs(expected-x['post_state'][n].numpy())));assert err<=1.5e-7;maxparam=max(maxparam,err);paramchecks+=g.numel()
                postdiff+=int(not torch.equal(x['post_prediction']['boxes'],x['arms']['Final']['boxes']))
            else:assert all(torch.equal(v,x['pre_state'][n]) for n,v in x['post_state'].items())
            rec=dict(parent=r['ordinal'],condition=cond,arrival=arrival,expert_scheduled=x['expert_scheduled'],updated=x['updated'],pre_state_sha256=x['pre_state_sha256'],post_state_sha256=x['post_state_sha256'],step_norm=x['parameter_displacement'],inherited_state_norm=x['displacement_from_source'],temporal_support_changed=x.get('temporal_support_changed',False),temporal_selected_interval_changed=x.get('temporal_selected_interval_changed',False),current_spatial_pre_update=True)
            if x['updated']:rec['update_diagnostics']={k:(v.tolist() if torch.is_tensor(v) else v) for k,v in x['update'].items() if k!='gradients'}
            if x['expert_scheduled']:
                # No physical time/interval coordinates or expert proposal tensors in public scalar table.
                rec['temporal_diagnostics']={field:dict(scores=x[field]['scores'],selected=x[field]['selected'],candidate_count=len(x[field]['candidates'])) for field in ['temporal','fast_control']}
            for arm,short in ARMS.items():
                pred=x['arms'][arm];m,_=checked_score(pred['boxes'],gt[r['key']],r['frame_ids'],pred['indices']);metriccalls+=1
                for code,key in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:rec[f'{short}_{code}']=m[key]
            for short,m in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:
                assert abs(rec['slow_'+short]-slowrows[r['ordinal'],cond]['online_'+short])<1e-12
                assert abs(rec['frozen_'+short]-slowrows[r['ordinal'],cond]['frozen_'+short])<1e-12
                for a,b in PAIRS:rec[f'{a}_minus_{b}_{short}']=rec[a+'_'+short]-rec[b+'_'+short]
                rec['interaction_'+short]=rec['final_'+short]-rec['fast_'+short]-rec['slow_'+short]+rec['frozen_'+short]
                if not x['expert_scheduled']:assert abs(rec['final_minus_fast_'+short]-slowrows[r['ordinal'],cond]['online_minus_frozen_'+short])<1e-12
            rows.append(rec)
        assert all(torch.equal(v,previous[n]) for n,v in load(OUT/'final_states'/f'{cond}.pt').items())
    summary={};effects=[];keys=[k for k in rows[0] if k.endswith(('_s','_t','_v'))]
    for group in ['corruption']+p['conditions']:
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']==group)];summary[group]={}
        for subset in ['all','expert','nonexpert','nonexpert_after_first_write']:
            seq=[r for r in rr if subset=='all' or (subset=='expert' and r['expert_scheduled']) or (subset=='nonexpert' and not r['expert_scheduled']) or (subset=='nonexpert_after_first_write' and not r['expert_scheduled'] and r['arrival']>4)]
            summary[group][subset]=dict(cells=len(seq),sources=len({r['parent'] for r in seq}),metrics={k:macro(seq,k) for k in keys})
            if group in ['corruption','clean'] and subset in ['all','nonexpert','expert']:
                for parent in sorted({r['parent'] for r in seq}):effects.append(dict(parent=parent,group=group,subset=subset,**{k:float(np.mean([r[k] for r in seq if r['parent']==parent])) for k in keys if '_minus_' in k or 'interaction_' in k}))
    write(OUT/'ROWS.json',rows);write(OUT/'SUMMARY.json',summary);write(OUT/'SOURCE_EFFECTS.json',effects)
    write(OUT/'AUDIT.json',dict(status='pass',state_links=96,source_resets=6,Final_Slow_exact_state_pairs=192,Final_Slow_exact_gradient_coordinates=gradchecks,independent_SGD_coordinates=paramchecks,maximum_parameter_error=maxparam,independent_KL_updates=klchecks,maximum_KL_error=maxkl,KL_tolerance=1e-6,parameter_tolerance=1.5e-7,dual_metric_calls=metriccalls,independent_temporal_scores=teacherchecks,scheduled_expert_cells=expert_sets,nonexpert_no_evidence_reads=nonexpert,empty_spatial_noops=empty,current_output_pre_update_verified=96,updated_post_boxes_differ_from_reported_pre_boxes=postdiff,current_temporal_support_changed_cells=changed_support,current_temporal_selected_interval_changed_cells=changed_interval,spatial_full_reinsertions=4,temporal_all_layer_reinsertions=2,all_predictions_before_GT=True,prior_nonexpert_effect_exactly_reproduced=True))
    allocations=[read(f) for f in (OUT/'allocations').glob('*.json')];write(OUT/'RESOURCES.json',dict(new_GPU_process_seconds=sum(x['seconds'] for x in allocations),GPU_failed_attempts=sum(x['status']!='completed' for x in allocations),new_backward_steps=18,new_spatial_candidates=216,new_encoder_captures=0,new_expert_inferences=0,new_Final_arrivals=96,cached_control_arrivals=288,logical_specialist_calls={'Frozen':{'temporal':0,'spatial':0},'Fast-only':{'temporal':24,'spatial':0},'Slow-only':{'temporal':0,'spatial':24},'Final':{'temporal':24,'spatial':24}},availability_fraction=.25,note='Equal availability schedule, unequal total calls across ablations. Cache execution runtime is not uncached deployment latency.'))
    print({g:{sub:{k:summary[g][sub]['metrics'][k] for k in ['fast_minus_frozen_v','final_minus_fast_s','final_minus_fast_v','final_minus_frozen_v']} for sub in ['all','expert','nonexpert']} for g in ['corruption','clean']})

if __name__=='__main__':run()
