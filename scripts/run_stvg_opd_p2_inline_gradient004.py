"""Bounded P2-only independent in-loop precision check and receipt dispatch."""
import gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_input_bridge003 import verify as previous_verify,run as previous_run
import scripts.run_stvg_opd_p2_engineering002 as previous
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,status,sha
from scripts.decota_matrix_common_v1 import load,save
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
REC=BASE/'recovery/P2_inline_gradient_precision_004'
ALLOCATOR=previous.ALLOCATOR


class QualificationComplete(Exception):pass


def verify():
    previous_verify();r=read(REC/'REVISION_RUNTIME.json')
    assert sha(previous.REC/'REVISION_RUNTIME.json')==r['original_engineering_runtime_sha256']
    assert sha(BASE/'recovery/P2_input_schema_003/REVISION_RUNTIME.json')==r['input_schema_runtime_sha256']
    assert sha(REC/'CAPTURE_RECEIPT.json')==r['capture_sha256']
    assert sha(REC/'REPRODUCTION_RECEIPT.json')==r['reproduction_sha256']
    assert sha(REC/'CPU_CONTRACTS.json')==r['CPU_contracts_sha256']
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    for f,rc in read(REC/'CAPTURE_RECEIPT.json')['originals'].items():
        if f.startswith(('scripts/','vg_tta/','protocols/','methods/')):
            assert sha(ROOT/f)==rc['sha256'],f
    import torch
    assert torch.version.git_version==r['torch_git_version']
    return r


def component_audit(fit,expert):
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import FIELD,REVISION,audit
    if FIELD not in fit:return ORIGINAL_COMPONENT_DISPATCH(fit,expert)
    checked=audit(fit,expert,previous.ORIGINAL_COMPONENT_AUDIT)
    return dict(checked,revision=REVISION,
        original_component_runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json'),
        original_engineering_runtime_sha256=sha(previous.REC/'REVISION_RUNTIME.json'),
        inline_precision_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),
        loss_reward_actions_gradients_Adam_and_scientific_configuration_unchanged=True)


ORIGINAL_COMPONENT_DISPATCH=previous.component_audit
ORIGINAL_SAVED_DISPATCH=previous.dispatch_saved
def dispatch_saved(fit,expert,z,binding=None):
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import REVISION,FIELD
    if z['math_audit'].get('revision')==REVISION:
        assert FIELD in fit and not fit.get('direct_objective')
        assert z['inline_precision_runtime_sha256']==sha(REC/'REVISION_RUNTIME.json')
        return component_audit(fit,expert)
    assert FIELD not in fit,'Unknown inline precision receipt must fail closed'
    return ORIGINAL_SAVED_DISPATCH(fit,expert,z,binding)


def partial_original_readback(fit):
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import FIELD
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
    trace=w['chart_context']['trace']
    equal(fit['numerical_chart']['trace'][:len(trace)],trace,counts)
    return dict(**counts,actual_saved_original_reproduction_prefix_and_failed_derivative_bitwise=True,
        original_dead_failed_fit_serialized=False,no_dead_memory_bitwise_claim=True)


def failure_qualification_or_resume(job,name):
    assert name.startswith('P2_')
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import Installation,FIELD,comparable
    from vg_tta.decota_spatial_opd_chart_revision003 import Installation as Chart
    from methods.decota_final_simplified_v1.tensors import state_hash
    import torch
    import scripts.run_stvg_opd_paper_components_v2 as runner
    qualifying=job=='GPU_requal';oldsave=runner.save;old_init=Chart.__init__
    inline=Installation();charts=[];seen=False
    def init(self,*args,**kwargs):
        old_init(self,*args,**kwargs);charts.append(self)
    Chart.__init__=init
    def checked_save(path,z):
        nonlocal seen
        if not str(path).startswith(str(BASE/'stages')) or not isinstance(z,dict) or 'fit' not in z:
            return oldsave(path,z)
        assert z['stage']==name and z['GT_read'] is False
        if not z['fit'].get('direct_objective'):
            assert FIELD in z['fit']
            z['inline_precision_runtime_sha256']=sha(REC/'REVISION_RUNTIME.json')
            z['compute']['inline_independent_CPU_seconds_in_fit_wall_time']=z['fit'][FIELD]['CPU_audit_seconds']
        first=(name=='P2_hc2_same_5percent' and z['condition']=='frame_drop_5'
               and z['order']=='order1' and z['arrival']==7 and z['arm']=='frozen_rollout')
        folder=REC/'qualification/failed_query'
        if first:
            seen=True;w=load(REC/'REPRODUCED_INCOMPLETE_FIT_WITNESS.pt')
            assert z['query_ordinal']==3051 and z['previous_payload_sha256']==w['previous_payload_sha256']
            assert z['input']['sha256']==w['input']['sha256'] and z['input']['new_DINO_calls']==0
            partial=partial_original_readback(z['fit'])
            if qualifying:
                assert not path.exists() and not folder.exists()
                frame=sys._getframe(1)
                while frame is not None and not frame.f_code.co_filename.endswith('run_stvg_opd_paper_components_v2.py'):
                    frame=frame.f_back
                assert frame is not None;loc=frame.f_locals
                torch.cuda.synchronize()
                repeated=loc['fit_variant'](loc['base'],loc['initial'],loc['ex'],loc['row']['frame_ids'],
                    loc['row']['key'],loc['arm'],loc['cfg'])
                torch.cuda.synchronize();counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                equal(comparable(z['fit'],True),comparable(repeated,True),counts)
                assert component_audit(repeated,loc['ex'])==z['math_audit']
                assert state_hash(loc['model'].state_dict())==loc['modelhash']
                assert state_hash(loc['expert'].model.state_dict())==loc['experthash']
                folder.mkdir(parents=True)
                save(folder/'QUALIFIED_COMPLETE_FIT.pt',z);save(folder/'REPEATED_COMPLETE_FIT.pt',repeated)
                write(folder/'GPU_QUALIFICATION.json',dict(status='pass',scope='two real complete unchanged 40-round frozen-rollout fits after independent in-loop check; original replay prefix equality',
                    stage=name,condition=z['condition'],order=z['order'],arrival=7,query_ordinal=3051,
                    actual_GPU_fits=2,original_reproduced_partial_fit_bitwise=partial,
                    repeated_complete_fit_bitwise=counts,
                    qualified_fit_sha256=sha(folder/'QUALIFIED_COMPLETE_FIT.pt'),
                    repeated_fit_sha256=sha(folder/'REPEATED_COMPLETE_FIT.pt'),
                    input_sha256=z['input']['sha256'],source_and_expert_unchanged=True,
                    new_DINO_calls=0,formal_predictions_accepted=0,GT_read=False,
                    runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
                raise QualificationComplete()
            qualified=load(folder/'QUALIFIED_COMPLETE_FIT.pt');counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            equal(comparable(z['fit'],True),comparable(qualified['fit'],True),counts)
            if not (REC/'FIRST_FORMAL_FIT_BITWISE.json').exists():
                write(REC/'FIRST_FORMAL_FIT_BITWISE.json',dict(status='pass',stage=name,condition=z['condition'],
                    order=z['order'],arrival=7,query_ordinal=3051,qualified_complete_fit_bitwise=counts,
                    original_reproduced_partial_fit_bitwise=partial,qualified_fit_sha256=sha(folder/'QUALIFIED_COMPLETE_FIT.pt'),
                    GT_read=False,next_operation='accept first missing formal fit only after actual qualification equality',time=time.time()))
        assert not qualifying or first,'Qualification cannot accept any new formal prediction'
        return oldsave(path,z)
    runner.save=checked_save
    try:
        previous_run('GPU',name)
        assert not qualifying,'Qualification must stop before saving a formal prediction'
    except QualificationComplete:
        assert qualifying and seen
        print('P2_INLINE004_FAILED_QUERY_TWO_COMPLETE_REAL_FITS_PASS_ZERO_FORMAL',flush=True)
    finally:
        Chart.__init__=old_init;runner.save=oldsave;inline.uninstall();gc.collect()


def controls():
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.stvg_opd_paper_inputs_v1 import capture
    from scripts.stvg_opd_paper_later_common_v1 import stage_definition
    from scripts.stvg_opd_legacy_input_bridge003 import install
    from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import Installation,comparable
    from vg_tta.decota_spatial_opd_chart_revision003 import Installation as Chart
    import vg_tta.stvg_opd_paper_component_audit_v1 as target
    import collections
    install();previous.ORIGINAL_COMPONENT_AUDIT=target.audit
    dest=REC/'qualification/ordinary_controls';assert not dest.exists();dest.mkdir(parents=True)
    records=[];lease=gpu()
    inline=Installation();chart=Chart()
    try:
        for name,condition in read(REC/'REVISION_RUNTIME.json')['ordinary_controls']:
            stage=stage_definition(name);model=model_for(stage['source']);mh=state_hash(model.state_dict())
            expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);eh=state_hash(expert.model.state_dict())
            cache=collections.OrderedDict();seq=stage['orders']['order1'][:2]
            for at,q in enumerate(seq):
                row=read_row(stage['dataset'],q)
                base,native,ex,inp=capture(model,expert,stage,row,condition,cache)
                assert inp['new_DINO_calls']==0
                for arm in ['on_policy','frozen_rollout','shuffled_feedback']:
                    oldpath=BASE/'component_qualification'/name/condition/'order1'/arm/f'{at:05}.pt'
                    old=load(oldpath);assert sha(BASE/old['input']['path'])==old['input']['sha256']==inp['sha256']
                    cfg=stage['variant_configs'][arm];initial=old['fit']['initial']
                    new=fit_variant(base,initial,ex,row['frame_ids'],row['key'],arm,cfg)
                    mathcheck=component_audit(new,ex)
                    new_function=chart.old_fit;chart.old_fit=inline.old_fit
                    try:prior=fit_variant(base,initial,ex,row['frame_ids'],row['key'],arm,cfg)
                    finally:chart.old_fit=new_function
                    bitwise=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                    equal(comparable(new),prior,bitwise)
                    historical=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                    equal(prior,old['fit'],historical)
                    assert ORIGINAL_COMPONENT_DISPATCH(prior,ex)==old['math_audit']
                    path=dest/(name+'_'+arm+f'_{at:05}.pt')
                    z=dict(old,fit=new,old_control_fit=prior,math_audit=mathcheck,
                           inline_precision_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'))
                    save(path,z)
                    records.append(dict(stage=name,condition=condition,arm=arm,arrival=at,query_ordinal=q,
                        path=str(path.relative_to(BASE)),sha256=sha(path),bytes=path.stat().st_size,
                        original_qualification_sha256=sha(oldpath),input_sha256=inp['sha256'],
                        old_new_complete_fit_bitwise=bitwise,prior_qualified_full_fit_bitwise=historical,
                        actual_GPU_fits=2,GT_read=False,new_DINO_calls=0,formal_predictions_accepted=0))
                    print('P2_INLINE004_ORDINARY_OLD_NEW_FULL_PASS',name,arm,at,flush=True)
                base.restore(base.initial);del base,native,ex;gc.collect();torch.cuda.empty_cache()
            assert state_hash(model.state_dict())==mh and state_hash(expert.model.state_dict())==eh
            del model,expert;gc.collect();torch.cuda.empty_cache()
        assert len(records)==12
        write(dest/'GPU_QUALIFICATION.json',dict(status='pass',scope='two predeclared ordinary arrivals in both original source configurations for all three Gaussian arms',
            actual_GPU_fits=24,qualification_cells=12,records=records,source_and_expert_unchanged=True,
            GT_read=False,new_DINO_calls=0,formal_predictions_accepted=0,
            runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
    finally:chart.uninstall();inline.uninstall();lease.close();gc.collect()


def root_qualification():
    import torch
    torch.set_num_threads(2)
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_paper_later_common_v1 import committed
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import comparable
    import vg_tta.stvg_opd_paper_component_audit_v1 as target
    previous.ORIGINAL_COMPONENT_AUDIT=target.audit
    fq=read(REC/'qualification/failed_query/GPU_QUALIFICATION.json')
    oc=read(REC/'qualification/ordinary_controls/GPU_QUALIFICATION.json')
    assert fq['status']==oc['status']=='pass' and fq['actual_GPU_fits']==2 and oc['actual_GPU_fits']==24
    folder=REC/'qualification/failed_query';z=load(folder/'QUALIFIED_COMPLETE_FIT.pt')
    repeated=load(folder/'REPEATED_COMPLETE_FIT.pt');counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
    equal(comparable(z['fit'],True),comparable(repeated,True),counts)
    partial=partial_original_readback(z['fit']);assert partial==fq['original_reproduced_partial_fit_bitwise']
    records=[dict(path=str((folder/'QUALIFIED_COMPLETE_FIT.pt').relative_to(BASE)),sha256=sha(folder/'QUALIFIED_COMPLETE_FIT.pt'))]+oc['records']
    paths=coordinates=rounds=0
    for rc in records:
        p=BASE/rc['path'];assert sha(p)==rc['sha256'];v=load(p);f=v['fit']
        assert sha(BASE/v['input']['path'])==v['input']['sha256']
        inp=load(BASE/v['input']['path']);expert=unpack_expert(inp['expert'])
        assert component_audit(f,expert)==v['math_audit']
        assert torch.count_nonzero(f['initial']['spatial.query_residual'])==0
        expected=committed(f['initial'],f['state'],v['config']['writeback'])
        assert all(torch.equal(expected[n],v['committed'][n]) for n in expected)
        if 'old_control_fit' in v:
            equal(comparable(f),v['old_control_fit'],dict(tensors=0,tensor_coordinates=0,scalar_values=0))
        paths+=1;rounds+=len(f['rounds']);coordinates+=sum(a.numel() for a in f['state'].values())
    assert paths==13
    write(REC/'ROOT_QUALIFICATION_READBACK.json',dict(status='pass',scope='actual independent complete qualification Gaussian/IoU/softmax/gradient/Adam/chart/1792/writeback/reset readback',
        actual_GPU_fits=26,complete_saved_fit_readbacks=paths,rounds_read_back=rounds,state_coordinates=coordinates,
        repeated_complete_fit_bitwise=counts,original_reproduced_partial_fit_bitwise=partial,
        failed_query_qualification_sha256=sha(folder/'GPU_QUALIFICATION.json'),
        ordinary_control_qualification_sha256=sha(REC/'qualification/ordinary_controls/GPU_QUALIFICATION.json'),
        formal_predictions_accepted=0,new_DINO_calls=0,GT_read=False,independent_full_decoder_Jacobian=False,
        runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
    write(REC/'GPU_QUALIFICATION.json',dict(status='pass',scope='bounded P2-only in-loop audit supplement qualification',
        actual_GPU_fits=26,ordinary_control_cells=12,root_readback_sha256=sha(REC/'ROOT_QUALIFICATION_READBACK.json'),
        failed_query_qualification_sha256=sha(folder/'GPU_QUALIFICATION.json'),
        formal_predictions_accepted=0,new_DINO_calls=0,GT_read=False,time=time.time()))
    print('P2_INLINE004_ACTUAL_ROOT_26_REAL_FITS_READBACK_PASS',flush=True)


def run(job,name='P2'):
    verify()
    from scripts.stvg_opd_legacy_input_bridge003 import install
    install()
    if job=='root_qualification':return root_qualification()
    if job=='GPU_controls':
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')==ALLOCATOR
        return controls()
    prior_component=previous.component_audit;prior_dispatch=previous.dispatch_saved
    previous.component_audit=component_audit;previous.dispatch_saved=dispatch_saved
    try:
        if job in ['GPU','GPU_requal']:
            assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')==ALLOCATOR
            if job=='GPU':assert read(REC/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
            return failure_qualification_or_resume(job,name)
        assert job in ['score','finalize'] and name=='P2'
        return previous_run(job,name)
    finally:previous.component_audit=prior_component;previous.dispatch_saved=prior_dispatch


if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException:
        p=REC/'new_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc());raise
