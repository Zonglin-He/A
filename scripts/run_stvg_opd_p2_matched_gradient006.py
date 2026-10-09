"""Bounded P2 CUDA-reduction-matched CPU audit and immutable receipt dispatch."""
import gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_stvg_opd_p2_action_reward005 as original5
prior=original5.prior
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,status,sha
from scripts.decota_matrix_common_v1 import load,save
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
from vg_tta.decota_spatial_opd_matched_gradient_precision006 import mode,FIELD,REVISION,audit
REC=BASE/'recovery/P2_matched_gradient_precision_006'
ALLOCATOR=original5.ALLOCATOR
ORIGINAL_COMPONENT5=original5.component_audit
ORIGINAL_SAVED5=original5.dispatch_saved

class QualificationComplete(Exception):pass

def verify():
    original5.verify();r=read(REC/'REVISION_RUNTIME.json')
    for path,key in [(original5.REC/'REVISION_RUNTIME.json','actual_action_runtime_sha256'),
        (REC/'CAPTURE_RECEIPT.json','capture_sha256'),(REC/'REPRODUCTION_RECEIPT.json','reproduction_sha256'),
        (REC/'CPU_CONTRACTS.json','CPU_contracts_sha256'),(REC/'CPU_KERNEL_DIAGNOSIS_SOURCE_PIN.json','source_pin_sha256')]:
        assert sha(path)==r[key]
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    for f,rc in read(REC/'CAPTURE_RECEIPT.json')['originals'].items():
        if f.startswith(('scripts/','vg_tta/','protocols/','methods/')):assert sha(ROOT/f)==rc['sha256'],f
    import torch
    assert torch.version.git_version==r['torch_git_version']
    return r

def component_audit(fit,expert):
    if fit.get(FIELD,{}).get('revision')!=REVISION:
        with mode(False):return ORIGINAL_COMPONENT5(fit,expert)
    checked=audit(fit,expert,prior.previous.ORIGINAL_COMPONENT_AUDIT)
    return dict(checked,revision=REVISION,
        original_component_runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json'),
        inline_precision_runtime_sha256=sha(prior.REC/'REVISION_RUNTIME.json'),
        actual_action_runtime_sha256=sha(original5.REC/'REVISION_RUNTIME.json'),
        matched_gradient_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),
        original_GPU_actions_rewards_loss_gradients_Adam_and_scientific_configuration_unchanged=True)

def dispatch_saved(fit,expert,z,binding=None):
    if z['math_audit'].get('revision')==REVISION:
        assert fit[FIELD]['revision']==REVISION and not fit.get('direct_objective')
        assert z['matched_gradient_runtime_sha256']==sha(REC/'REVISION_RUNTIME.json')
        assert z['actual_action_runtime_sha256']==sha(original5.REC/'REVISION_RUNTIME.json')
        assert z['inline_precision_runtime_sha256']==sha(prior.REC/'REVISION_RUNTIME.json')
        return component_audit(fit,expert)
    assert fit.get(FIELD,{}).get('revision')!=REVISION,'Unknown matched-gradient receipt must fail closed'
    with mode(False):return ORIGINAL_SAVED5(fit,expert,z,binding)

def partial_original_readback(fit):
    w=load(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt');k=w['failed_round']
    counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
    equal(fit['initial'],w['initial'],counts)
    for i,h in enumerate(w['incomplete_path']):
        current=fit['path'][i]
        if i==k:current={n:v for n,v in current.items() if n!='update'}
        equal(current,h,counts)
    equal(fit['rounds'][:k],w['completed_rounds'],counts)
    r=fit['rounds'][k]
    equal(r['mean_before'],w['mean'],counts);equal(r['rollout'],w['rollout'],counts)
    equal(r['autograd_mean_gradient'],w['autograd_mean_gradient'],counts)
    equal(fit[FIELD]['rounds'][k]['analytic32'],w['analytic_mean_gradient'],counts)
    assert r['analytic_gradient_error']==w['original_inline_error']
    equal(fit['numerical_chart']['trace'][:len(w['chart_context']['trace'])],w['chart_context']['trace'],counts)
    from vg_tta.decota_spatial_opd_action_readback_precision005 import FIELD as ACTION
    equal(fit[ACTION]['trace'],w['actual_action_context']['trace'],counts)
    # Original004 dictionaries remain independently recomputable for completed rounds.
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import check as dynamically_bound
    from vg_tta.decota_spatial_opd_matched_gradient_precision006 import ORIGINAL_CHECK
    for i in range(k):
        old=w['inline_context']['rounds'][i];round_=fit['rounds'][i];rr=round_['rollout']
        checked=ORIGINAL_CHECK(round_['mean_before'].numpy(),rr['mean'].numpy(),rr['samples'].numpy(),
            rr['weights'].numpy(),round_['autograd_mean_gradient'].numpy(),old['analytic32'].numpy(),
            fit['config']['sigma'],round_['analytic_gradient_error'])
        assert checked==old['check']
    assert not fit[FIELD]['rounds'][k]['check']['original004_CPU32_absolute_guard_passed']
    return dict(**counts,actual_saved_original_reproduction_prefix_and_failed_derivative_bitwise=True,
        original_dead_failed_fit_serialized=False,no_dead_memory_bitwise_claim=True)

def failure_qualification_or_resume(job,name):
    import torch
    from vg_tta.decota_spatial_opd_action_readback_precision005 import comparable
    from methods.decota_final_simplified_v1.tensors import state_hash
    import scripts.run_stvg_opd_paper_components_v2 as runner
    qualifying=job=='GPU_requal';oldsave=runner.save;seen=False
    def checked_save(path,z):
        nonlocal seen
        if not str(path).startswith(str(BASE/'stages')) or not isinstance(z,dict) or 'fit' not in z:return oldsave(path,z)
        assert z['stage']==name and z['GT_read'] is False
        if not z['fit'].get('direct_objective'):
            assert z['fit'][FIELD]['revision']==REVISION
            z['matched_gradient_runtime_sha256']=sha(REC/'REVISION_RUNTIME.json')
        first=(name=='P2_hc2_same_5percent' and z['condition']=='exposure_5' and z['order']=='order1'
            and z['arrival']==26 and z['arm']=='frozen_rollout')
        folder=REC/'qualification/failed_query'
        if first:
            seen=True;w=load(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt')
            assert z['query_ordinal']==1151 and z['previous_payload_sha256']==w['previous_payload_sha256']
            assert z['input']['sha256']==w['input']['sha256'] and z['input']['new_DINO_calls']==0
            partial=partial_original_readback(z['fit'])
            if qualifying:
                assert not path.exists() and not folder.exists()
                frame=sys._getframe(1)
                while frame is not None and not frame.f_code.co_filename.endswith('run_stvg_opd_paper_components_v2.py'):frame=frame.f_back
                assert frame is not None;loc=frame.f_locals;torch.cuda.synchronize()
                repeated=loc['fit_variant'](loc['base'],loc['initial'],loc['ex'],loc['row']['frame_ids'],loc['row']['key'],loc['arm'],loc['cfg'])
                torch.cuda.synchronize();counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                equal(comparable(z['fit'],True),comparable(repeated,True),counts)
                assert component_audit(repeated,loc['ex'])==z['math_audit']
                assert state_hash(loc['model'].state_dict())==loc['modelhash']
                assert state_hash(loc['expert'].model.state_dict())==loc['experthash']
                folder.mkdir(parents=True);save(folder/'QUALIFIED_COMPLETE_FIT.pt',z);save(folder/'REPEATED_COMPLETE_FIT.pt',repeated)
                write(folder/'GPU_QUALIFICATION.json',dict(status='pass',scope='two real complete unchanged 40-round frozen-rollout fits with pinned CUDA reduction CPU32 audit and actual original incomplete replay equality',
                    stage=name,condition=z['condition'],order=z['order'],arrival=26,query_ordinal=1151,
                    actual_GPU_fits=2,original_reproduced_partial_fit_bitwise=partial,repeated_complete_fit_and_checks_bitwise=counts,
                    qualified_fit_sha256=sha(folder/'QUALIFIED_COMPLETE_FIT.pt'),repeated_fit_sha256=sha(folder/'REPEATED_COMPLETE_FIT.pt'),
                    input_sha256=z['input']['sha256'],source_and_expert_unchanged=True,new_DINO_calls=0,
                    formal_predictions_accepted=0,GT_read=False,runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
                raise QualificationComplete()
            qualified=load(folder/'QUALIFIED_COMPLETE_FIT.pt');counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            equal(comparable(z['fit'],True),comparable(qualified['fit'],True),counts)
            if not (REC/'FIRST_FORMAL_FIT_BITWISE.json').exists():
                write(REC/'FIRST_FORMAL_FIT_BITWISE.json',dict(status='pass',stage=name,condition=z['condition'],order=z['order'],arrival=26,query_ordinal=1151,
                    qualified_complete_fit_and_checks_bitwise=counts,original_reproduced_partial_fit_bitwise=partial,
                    qualified_fit_sha256=sha(folder/'QUALIFIED_COMPLETE_FIT.pt'),GT_read=False,
                    next_operation='accept first missing formal fit only after actual qualification equality',time=time.time()))
        assert not qualifying or first,'Qualification cannot accept any new formal prediction'
        return oldsave(path,z)
    runner.save=checked_save
    try:
        with mode():original5.run('GPU',name)
        assert not qualifying,'Qualification must stop before formal acceptance'
    except QualificationComplete:
        assert qualifying and seen;print('P2_MATCHED006_FAILED_QUERY_TWO_COMPLETE_REAL_FITS_PASS_ZERO_FORMAL',flush=True)
    finally:runner.save=oldsave;gc.collect()


def controls():
    import torch,collections
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.stvg_opd_paper_inputs_v1 import capture
    from scripts.stvg_opd_paper_later_common_v1 import stage_definition
    from scripts.stvg_opd_legacy_input_bridge003 import install
    from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
    from vg_tta.decota_spatial_opd_action_readback_precision005 import Installation,comparable
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import Installation as Inline
    from vg_tta.decota_spatial_opd_chart_revision003 import Installation as Chart
    import vg_tta.stvg_opd_paper_component_audit_v1 as target
    install();prior.previous.ORIGINAL_COMPONENT_AUDIT=target.audit
    dest=REC/'qualification/ordinary_controls';assert not dest.exists();dest.mkdir(parents=True)
    records=[];lease=gpu();actions=Installation();inline=Inline();chart=Chart()
    try:
        for name,condition in read(REC/'REVISION_RUNTIME.json')['ordinary_controls']:
            stage=stage_definition(name);model=model_for(stage['source']);mh=state_hash(model.state_dict())
            expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);eh=state_hash(expert.model.state_dict())
            cache=collections.OrderedDict();seq=stage['orders']['order1'][:2]
            for at,q in enumerate(seq):
                row=read_row(stage['dataset'],q);base,native,ex,inp=capture(model,expert,stage,row,condition,cache)
                assert inp['new_DINO_calls']==0
                for arm in ['on_policy','frozen_rollout','shuffled_feedback']:
                    oldpath=BASE/'component_qualification'/name/condition/'order1'/arm/f'{at:05}.pt'
                    old=load(oldpath);assert sha(BASE/old['input']['path'])==old['input']['sha256']==inp['sha256']
                    cfg=stage['variant_configs'][arm];initial=old['fit']['initial']
                    new=fit_variant(base,initial,ex,row['frame_ids'],row['key'],arm,cfg);mathcheck=component_audit(new,ex)
                    new_function=chart.old_fit;new_action=actions.core.action_boxes
                    chart.old_fit=inline.old_fit;actions.core.action_boxes=actions.old_action
                    try:old_control=fit_variant(base,initial,ex,row['frame_ids'],row['key'],arm,cfg)
                    finally:chart.old_fit=new_function;actions.core.action_boxes=new_action
                    bitwise=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                    equal(comparable(new),old_control,bitwise)
                    historical=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                    equal(old_control,old['fit'],historical)
                    assert component_audit(old_control,ex)==old['math_audit']
                    path=dest/(name+'_'+arm+f'_{at:05}.pt')
                    z=dict(old,fit=new,old_control_fit=old_control,math_audit=mathcheck,
                        inline_precision_runtime_sha256=sha(prior.REC/'REVISION_RUNTIME.json'),
                        actual_action_runtime_sha256=sha(original5.REC/'REVISION_RUNTIME.json'),
                        matched_gradient_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'))
                    save(path,z)
                    records.append(dict(stage=name,condition=condition,arm=arm,arrival=at,query_ordinal=q,
                        path=str(path.relative_to(BASE)),sha256=sha(path),bytes=path.stat().st_size,
                        original_qualification_sha256=sha(oldpath),input_sha256=inp['sha256'],
                        old_new_complete_fit_bitwise=bitwise,prior_qualified_full_fit_bitwise=historical,
                        actual_GPU_fits=2,GT_read=False,new_DINO_calls=0,formal_predictions_accepted=0))
                    print('P2_MATCHED006_ORDINARY_OLD_NEW_FULL_PASS',name,arm,at,flush=True)
                base.restore(base.initial);del base,native,ex;gc.collect();torch.cuda.empty_cache()
            assert state_hash(model.state_dict())==mh and state_hash(expert.model.state_dict())==eh
            del model,expert;gc.collect();torch.cuda.empty_cache()
        assert len(records)==12
        write(dest/'GPU_QUALIFICATION.json',dict(status='pass',scope='two predeclared ordinary arrivals in both original sources for all three Gaussian arms',
            actual_GPU_fits=24,qualification_cells=12,records=records,source_and_expert_unchanged=True,
            GT_read=False,new_DINO_calls=0,formal_predictions_accepted=0,
            runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
    finally:chart.uninstall();inline.uninstall();actions.uninstall();lease.close();gc.collect()


def root_qualification():
    import torch
    torch.set_num_threads(2)
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_paper_later_common_v1 import committed
    from vg_tta.decota_spatial_opd_action_readback_precision005 import comparable
    import vg_tta.stvg_opd_paper_component_audit_v1 as target
    prior.previous.ORIGINAL_COMPONENT_AUDIT=target.audit
    folder=REC/'qualification/failed_query';fq=read(folder/'GPU_QUALIFICATION.json')
    oc=read(REC/'qualification/ordinary_controls/GPU_QUALIFICATION.json')
    assert fq['status']==oc['status']=='pass' and fq['actual_GPU_fits']==2 and oc['actual_GPU_fits']==24
    z=load(folder/'QUALIFIED_COMPLETE_FIT.pt');repeated=load(folder/'REPEATED_COMPLETE_FIT.pt')
    counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
    equal(comparable(z['fit'],True),comparable(repeated,True),counts);partial=partial_original_readback(z['fit'])
    records=[dict(path=str((folder/'QUALIFIED_COMPLETE_FIT.pt').relative_to(BASE)),sha256=sha(folder/'QUALIFIED_COMPLETE_FIT.pt'))]+oc['records']
    paths=coordinates=rounds=oldguardfails=0
    for rc in records:
        p=BASE/rc['path'];assert sha(p)==rc['sha256'];v=load(p);f=v['fit']
        assert sha(BASE/v['input']['path'])==v['input']['sha256'];inp=load(BASE/v['input']['path']);expert=unpack_expert(inp['expert'])
        assert component_audit(f,expert)==v['math_audit']
        assert torch.count_nonzero(f['initial']['spatial.query_residual'])==0
        expected=committed(f['initial'],f['state'],v['config']['writeback'])
        assert all(torch.equal(expected[n],v['committed'][n]) for n in expected)
        if 'old_control_fit' in v:equal(comparable(f),v['old_control_fit'],dict(tensors=0,tensor_coordinates=0,scalar_values=0))
        paths+=1;rounds+=len(f['rounds']);coordinates+=sum(a.numel() for a in f['state'].values())
        oldguardfails+=v['math_audit']['original004_CPU32_guard_failures']
    assert paths==13 and oldguardfails>0
    write(REC/'ROOT_QUALIFICATION_READBACK.json',dict(status='pass',scope='actual independent complete qualification Gaussian/original-action-IoU/softmax/gradient/Adam/chart/1792/writeback/reset readback',
        actual_GPU_fits=26,complete_saved_fit_readbacks=paths,rounds_read_back=rounds,state_coordinates=coordinates,
        repeated_complete_fit_and_checks_bitwise=counts,original_reproduced_partial_fit_bitwise=partial,
        original004_CPU32_guard_failures_retained=oldguardfails,failed_query_qualification_sha256=sha(folder/'GPU_QUALIFICATION.json'),
        ordinary_control_qualification_sha256=sha(REC/'qualification/ordinary_controls/GPU_QUALIFICATION.json'),
        formal_predictions_accepted=0,new_DINO_calls=0,GT_read=False,independent_full_decoder_Jacobian=False,
        CUDA_transcendental_kernel_not_independently_proved=True,runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
    write(REC/'GPU_QUALIFICATION.json',dict(status='pass',scope='bounded P2-only pinned CUDA gradient reduction audit qualification',
        actual_GPU_fits=26,ordinary_control_cells=12,root_readback_sha256=sha(REC/'ROOT_QUALIFICATION_READBACK.json'),
        failed_query_qualification_sha256=sha(folder/'GPU_QUALIFICATION.json'),formal_predictions_accepted=0,new_DINO_calls=0,GT_read=False,time=time.time()))
    print('P2_MATCHED006_ACTUAL_ROOT_26_REAL_FITS_READBACK_PASS',flush=True)

def run(job,name='P2'):
    verify()
    from scripts.stvg_opd_legacy_input_bridge003 import install
    install()
    if job=='root_qualification':return root_qualification()
    if job=='GPU_controls':
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')==ALLOCATOR
        with mode():return controls()
    pc,pd=original5.component_audit,original5.dispatch_saved
    original5.component_audit=component_audit;original5.dispatch_saved=dispatch_saved
    try:
        if job in ['GPU','GPU_requal']:
            assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')==ALLOCATOR
            if job=='GPU':assert read(REC/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
            return failure_qualification_or_resume(job,name)
        assert job in ['score','finalize'] and name=='P2'
        return original5.run(job,name)
    finally:original5.component_audit,original5.dispatch_saved=pc,pd

if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException:
        p=REC/'new_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc());raise
