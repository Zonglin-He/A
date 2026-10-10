"""Actual bounded P5 root: complete population, every immutable fit and dense readout."""
import collections
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,read,write,sha
activate()
from scripts.stvg_opd_paper_later_common_v1 import phases,stage_definition,record_path,committed
from scripts.decota_matrix_common_v1 import load,status
from scripts.audit_stvg_opd_p5_public_v2 import audit as population
from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
from scripts.run_stvg_opd_p5_budget_precision001 import verify,dispatch_saved
DEST=BASE/'P5_actual_root_readback_revision001'


def progress(stage,**values):
    status(DEST/'STATUS.json',dict(status='running',scope='P5 complete postseal root CPU readback',stage=stage,
        worker_pid=os.getpid(),CPU_only=True,GT_after_global_seal=True,new_model_calls=0,paper_suite_complete=False,time=time.time(),**values))


def run():
    import numpy as np
    import torch
    torch.set_num_threads(2);verify();runtime=read(DEST/'RUNTIME.json')
    for f,h in runtime['pins'].items():assert sha(ROOT/f)==h,f
    seal=read(BASE/'P5_PREDICTION_BARRIER.json')
    assert seal['status']=='sealed' and seal['all_deployment_arms_and_directions']
    assert read(BASE/'P5_CPU_COMPLETION.json')['anonymous_rows']==2560
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import unpack_expert
    started=time.time();progress('complete_population_statistics')
    checked,data,pool=population(PUB/'P5')
    population_path=PUB/'P5/ACTUAL_ROOT_POPULATION_READBACK.json'
    population_receipt=dict(checked,all_sources_read=True)
    if population_path.exists():
        prior_population=read(population_path);prior_population.pop('time')
        assert prior_population==population_receipt
    else:write(population_path,dict(population_receipt,time=time.time()))
    from scripts.audit_stvg_opd_p5_public_v2 import budget_analysis
    write(PUB/'P5/ACTUAL_ROOT_BUDGET_CONTRASTS.json',budget_analysis(data))
    count=collections.Counter();revisions=collections.Counter();maximum=0.;records=[];sources={};feedback=[]
    for name in phases()['P5']:
        stage=stage_definition(name);ds=stage['dataset'];path=BASE/'stages'/name/'PREDICTION_BARRIER.json'
        assert sha(path)==seal['barriers'][str(path.relative_to(BASE))];barrier=read(path)
        assert barrier['orders']==stage['orders'] and barrier['variant_configs']==stage['variant_configs']
        assert barrier['status']=='sealed' and barrier['GT_read'] is False
        truth,spans,provenance=truths(ds,stage)
        plan=read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')
        ids={s:i for i,s in enumerate(sorted({r['source'] for r in plan['rows']}))}
        sources[name]=sorted({ids[plan['rows'][q]['source']] for seq in stage['orders'].values() for q in seq})
        assert len(sources[name])==128
        assert stage['arms']==['on_policy'] and list(stage['orders'])==['order1','order2'] and stage['conditions']==['clean']
        scored={(r['condition'],r['order'],r['arm'],r['arrival']):r for r in data[name]}
        origin_states={}
        for condition in stage['conditions']:
            for order,seq in stage['orders'].items():
                last={a:None for a in stage['arms']};predecessor={a:None for a in stage['arms']}
                for at,q in enumerate(seq):
                    common=None; row=plan['rows'][q]
                    for arm in stage['arms']:
                        p,binding=record_path(barrier,condition,order,arm,at);rc=read(p.with_suffix('.json'));h=sha(p)
                        assert h==binding['sha256']==rc['sha256'] and rc['bytes']==p.stat().st_size
                        assert not rc['GT_read'] and rc['time']<=barrier['time']<=seal['time']
                        z=load(p);fit=z['fit'];cfg=stage['variant_configs'][arm];saved=scored[condition,order,arm,at]
                        assert z['query_ordinal']==saved['query_ordinal']==q and saved['source_id']==ids[row['source']]
                        assert z['dataset']==ds and z['source']==stage['source'] and z['arm']==arm
                        assert z['condition']==condition and z['order']==order and z['arrival']==at and not z['GT_read']
                        assert z['config']==fit['config']==saved['config']==cfg and fit['selected_step']==cfg['steps']
                        assert z['previous_payload_sha256']==predecessor[arm] and fit['active_parameters']=={'query_only':256,'LN_only':1536}.get(arm,1792)
                        assert z['query_reset'] and z['Adam_reset'] and z['Native_WHEN_fixed']
                        assert sum(v.numel() for v in fit['initial'].values())==1792
                        assert torch.count_nonzero(fit['initial']['spatial.query_residual'])==0
                        if last[arm] is None:
                            if arm not in origin_states:origin_states[arm]={n:v.clone() for n,v in fit['initial'].items()}
                            assert all(torch.equal(v,origin_states[arm][n]) for n,v in fit['initial'].items())
                            qualified=load(BASE/'component_qualification'/name/condition/'order1'/arm/'00000.pt')['fit']['initial']
                            assert all(torch.equal(v,qualified[n]) for n,v in fit['initial'].items())
                        else:assert all(torch.equal(v,torch.zeros_like(v) if n=='spatial.query_residual' else last[arm][n]) for n,v in fit['initial'].items())
                        expected=committed(fit['initial'],fit['state'],cfg['writeback'])
                        assert all(torch.equal(expected[n],z['committed'][n]) for n in expected)
                        ip=BASE/z['input']['path'];ir=read(ip.with_suffix('.json'))
                        assert sha(ip)==z['input']['sha256']==barrier['inputs'][z['input']['path']]==ir['sha256']
                        assert not ir['GT_read'] and ir['time']<=barrier['time']
                        if 'bytes' in ir:assert ir['bytes']==ip.stat().st_size
                        else:
                            assert set(ir)=={'sha256','runtime_lock_sha256','GT_read','time'}
                            assert ir['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
                            original=read(BASE/'stages'/('P0_'+ds)/'PREDICTION_BARRIER.json')
                            original_prediction=BASE/'stages'/('P0_'+ds)/'clean'/order/arm/(p.stem+'.pt')
                            original_record=load(original_prediction)
                            assert z['input']==original_record['input']
                            assert original['inputs'][z['input']['path']]==z['input']['sha256']==ir['sha256']
                            count['strict_original_P0_input_receipt_readbacks']+=1
                        inp=load(ip)
                        count['input_bindings_checked']+=1
                        assert inp['GT_read'] is False and inp['frame_ids']==row['frame_ids'] and z['interval']==inp['interval']
                        identity=tuple(inp['frame_ids']),inp['pixel_sha256'],inp['interval'],inp['native_boxes'],inp['expert']
                        if common is None:common=identity
                        else:
                            from scripts.stvg_opd_paper_common_v1 import digest
                            assert common[:3]==identity[:3] and torch.equal(common[3],identity[3]) and digest(common[4])==digest(identity[4])
                            count['matched_input_checks']+=1
                        expert=unpack_expert(inp['expert']);assert dispatch_saved(fit,expert,z,binding)==z['math_audit']
                        if at<2 and order=='order1':
                            from scripts.run_stvg_opd_p5_budget_precision001 import scientific,REC
                            from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
                            qp=BASE/'component_qualification'/name/condition/order/arm/(p.stem+'.pt')
                            qz=load(qp);comparison=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                            assert qz['config']==cfg and qz['input']['sha256']==z['input']['sha256']
                            equal(scientific(fit),scientific(qz['fit']),comparison)
                            if binding['reused_complete_identical_stream']:
                                assert name.endswith('_K4') and binding['origin_stage']=='P0_'+ds
                                count['qualified_original_alias_pairs_bitwise']+=1
                            else:
                                cr=read(REC/'first_formal'/name/condition/order/arm/(p.stem+'.json'))
                                assert cr['status']=='pass' and cr['qualified_fit_sha256']==sha(qp) and not cr['GT_read']
                                assert comparison==cr['complete_fit_bitwise']
                                count['qualified_new_formal_pairs_bitwise']+=1
                            count['actual_qualified_formal_pairs_bitwise']+=1
                        revisions[str(z['math_audit'].get('revision'))]+=1;count['complete_math_dictionary_exact']+=1
                        curves={}
                        for label,boxes in [('Frozen',inp['native_boxes']),('Before',fit['before']),('After',fit['final'])]:
                            values,fids,iou=dense_metric(boxes.numpy(),row,truth[q],spans[q],z['interval'],ds);curves[label]=iou
                            for m,v in values.items():
                                error=abs(v-saved[label+'_'+m]);maximum=max(maximum,error);assert error<2e-10
                                count['dense_metric_scalars']+=1
                        observed=np.isin(fids,[row['frame_ids'][i] for i in fit['positions']]);difference=curves['After']-curves['Before']
                        for kind,mask in [('observed',observed),('unobserved',~observed)]:
                            value=float(difference[mask].mean()) if mask.any() else None;old=saved[kind+'_iou_delta']
                            assert (value is None and old is None) or (value is not None and old is not None and abs(value-old)<2e-10)
                            count['observed_unobserved_checks']+=1
                        signal=None
                        if arm=='on_policy' and fit['positions']:
                            valid=[j for j,pos in enumerate(fit['positions']) if row['frame_ids'][pos] in truth[q]]
                            if valid:
                                rd=fit['rounds'];signal=dict(central_expert_reward_delta=float(np.mean([
                                    (r['central_reward_after']-r['central_reward_before']).numpy()[valid].mean() for r in rd])),
                                    teacher_ESS=float(np.mean([r['ESS'].numpy()[valid].mean() for r in rd])),
                                    teacher_max_weight=float(np.mean([r['weight_max'].numpy()[valid].mean() for r in rd])),
                                    GT_evaluable_admitted_positions=len(valid))
                        if arm=='on_policy':feedback.append(dict(saved,feedback=signal))
                        last[arm]=z['committed'];predecessor[arm]=h
                        count['new_formal_fits']+=int(not binding['reused_complete_identical_stream'])
                        count['logical_arrivals']+=1;count['state_coordinates']+=sum(v.numel() for v in fit['state'].values())
                        count['rounds']+=len(fit['rounds']);count['payload_bytes']+=rc['bytes']
                        count['exact_stream_reuse_rows']+=int(binding['reused_complete_identical_stream'])
                        records.append(dict(stage=name,condition=condition,order=order,arm=arm,arrival=at,query_ordinal=q,
                            path=str(p.relative_to(BASE)),sha256=h,bytes=rc['bytes'],receipt_sha256=sha(p.with_suffix('.json')),
                            input_path=str(ip.relative_to(BASE)),input_sha256=z['input']['sha256'],input_receipt_sha256=sha(ip.with_suffix('.json'))))
                    if at%20==0:
                        progress('complete_math_state_dense',stage_name=name,condition=condition,order=order,arrival=at,done=count['logical_arrivals'],total=2560)
                        print('ROOT_P5_COMPLETE_READBACK',name,condition,order,at,count['logical_arrivals'],2560,flush=True)
        write(DEST/('COMPLETE_'+name+'.json'),dict(status='pass',scope='all stage fits/state/input/dense',
            stage=name,counts=dict(count),GT_provenance=provenance,time=time.time()))
    assert count['logical_arrivals']==count['complete_math_dictionary_exact']==2560
    assert count['exact_stream_reuse_rows']==512
    assert count['new_formal_fits']==2048
    assert count['input_bindings_checked']==2560 and count['actual_qualified_formal_pairs_bitwise']==20
    assert count['qualified_new_formal_pairs_bitwise']==16 and count['qualified_original_alias_pairs_bitwise']==4
    assert count['state_coordinates']==2560*1792 and count['rounds']==1024*40+1536*10
    progress('second_opaque_hash_readback',done=2560,total=2560)
    for r in records:
        assert sha(BASE/r['path'])==r['sha256'] and sha((BASE/r['path']).with_suffix('.json'))==r['receipt_sha256']
        assert sha(BASE/r['input_path'])==r['input_sha256'] and sha((BASE/r['input_path']).with_suffix('.json'))==r['input_receipt_sha256']
        count['second_opaque_prediction_input_receipt_checks']+=4
    write(PUB/'P5/ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json',dict(status='pass',counts=dict(count),
        maximum_dense_error=maximum,saved_audit_revision_counts=dict(revisions),records=records,
        all_original_reset_inheritance_writeback_and_Adam_checked=True,all_input_bindings_checked=True,all_condition_source_resets_checked=True,
        GT_after_global_seal=True,GPU_model_optimizer_calls=0,independent_full_decoder_Jacobian=False,
        immutable_predictions_and_receipts=True,CPU_seconds=time.time()-started,time=time.time()))
    write(PUB/'P5/ACTUAL_ROOT_FEEDBACK_ROWS.json',dict(status='pass',rows=feedback,
        conditioning='all Full rows; reward/ESS only GT-evaluable admitted positions; posthoc descriptive',time=time.time()))
    selected=[]
    for ds in ['hc2','vidstg']:
        eligible=[r for r in feedback if r['dataset']==ds and r['feedback'] and not r['stage'].startswith('P5_unified_')]
        used=set()
        for kind,group in [('success',sorted(eligible,key=lambda r:(-r['delta_current_v'],r['query_ordinal'],r['condition'],r['order'],r['stage']))),
            ('current_harm',sorted(eligible,key=lambda r:(r['delta_current_v'],r['query_ordinal'],r['condition'],r['order'],r['stage']))),
            ('expert_reward_task_mismatch',sorted([r for r in eligible if r['delta_current_v']<0 and r['feedback']['central_expert_reward_delta']>0],key=lambda r:(r['delta_current_v'],r['query_ordinal'],r['condition'],r['order'],r['stage'])))]:
            r=next(v for v in group if v['query_ordinal'] not in used);used.add(r['query_ordinal'])
            selected.append(dict(dataset=ds,stage=r['stage'],condition=r['condition'],arm=r['arm'],kind=kind,
                query_ordinal=r['query_ordinal'],source_id=r['source_id'],order=r['order'],arrival=r['arrival'],
                delta_current_v=r['delta_current_v'],delta_inherited_v=r['delta_inherited_v'],
                expert_reward_delta=r['feedback']['central_expert_reward_delta']))
    write(PUB/'P5/ACTUAL_ROOT_CASE_SELECTION.json',dict(status='posthoc_descriptive_cases_after_complete_population',records=selected,
        cases_are_not_efficacy_estimates=True,no_model_or_configuration_selection=True,time=time.time()))
    write(DEST/'COMPLETION.json',dict(status='pending_actual_root_signal_cases_view_publication',scope='P5 full source/math/state/dense root readback',
        actual_arrivals=2560,parent_sources_across_datasets=256,counts=dict(count),CPU_only=True,model_calls=0,
        paper_suite_complete=False,outputs={str(p.relative_to(ROOT)):sha(p) for p in (PUB/'P5').glob('ACTUAL_ROOT*.json')},time=time.time()))
    progress('pending_actual_root_signal_cases_view_publication',done=2560,total=2560)


if __name__=='__main__':
    try:run()
    except BaseException:
        p=DEST/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(DEST/'STATUS.json',dict(status='failed_preserved',evidence=str(p.relative_to(ROOT)),CPU_only=True,paper_suite_complete=False,time=time.time()));raise
