"""Original fixed P2 runner with separately pinned, already qualified P1 numerics.

Only the original joint Gaussian arms receive chart003 / audit001,002,005.
Direct L1/GIoU keeps its original fitter and audit. All old complete aliases
are read with their saved audit revision; no receipt or scientific lock changes.
"""
import gc, os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,activate,verify,read,write,status,sha
from scripts.decota_matrix_common_v1 import load,save
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
REC=BASE/'P2_engineering_revision002'
ALLOCATOR='expandable_segments:True'
REVISION='P2_joint_component_precision_revision002'


def verify_revision():
    verify();activate()
    from scripts.stvg_opd_paper_later_common_v1 import verify_later
    verify_later();r=read(REC/'REVISION_RUNTIME.json')
    assert sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json')==r['original_component_runtime_sha256']
    assert sha(BASE/'LATER_DESIGN_LOCK.json')==r['original_later_design_sha256']
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    assert read(BASE/'P1_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    assert read(BASE/'P1_FINAL_GITHUB_RECEIPT.json')['status']=='pass'
    return r


def old_receipt_audit(fit,expert,revision):
    from vg_tta.decota_spatial_opd_precision_audit_revision001 import audit as a1,original_audit
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import audit as a2
    from vg_tta.decota_spatial_opd_chart_revision003 import audit as a3,REVISION as r3
    from vg_tta.decota_spatial_opd_reward_audit_revision005 import audit as a5,REVISION as r5
    mapping={None:original_audit,'precision_audit_revision001':a1,'precision_audit_revision002':a2,r3:a3,r5:a5}
    assert revision in mapping,('Unregistered old audit revision',revision)
    return mapping[revision](fit,expert)


def component_audit(fit,expert):
    # The installed wrapper keeps this immutable function before monkeypatching.
    original=ORIGINAL_COMPONENT_AUDIT
    if fit.get('direct_objective'):return original(fit,expert)
    assert fit['active_parameters']==1792 and not fit.get('direct_objective')
    assert fit['arm'] in ['on_policy','frozen_rollout','shuffled_feedback']
    original_passed=True;failed_line=None
    try:original(fit,expert)
    except AssertionError as exc:
        tb=exc.__traceback__
        while tb.tb_next:tb=tb.tb_next
        assert tb.tb_frame.f_code.co_filename.endswith('stvg_opd_paper_component_audit_v1.py')
        assert tb.tb_lineno in (36,40,43),'Only registered mixed-precision reward/softmax/gradient checks may be supplemented'
        original_passed=False;failed_line=tb.tb_lineno
    from vg_tta.decota_spatial_opd_reward_audit_revision005 import audit
    checked=audit(fit,expert)
    return dict(checked,revision=REVISION,source_audit_revision=checked.get('revision'),
        original_component_absolute_audit_passed=original_passed,original_component_failed_line=failed_line,
        original_component_runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json'),
        engineering_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),
        loss_reward_actions_gradients_Adam_and_scientific_configuration_unchanged=True)


def dispatch_saved(fit,expert,z,binding=None):
    revision=z['math_audit'].get('revision')
    if binding is not None and binding.get('origin_stage','').startswith('P0_'):
        return old_receipt_audit(fit,expert,revision)
    if revision==REVISION:return component_audit(fit,expert)
    assert revision is None and fit.get('direct_objective'),'Unknown P2 audit receipt must fail closed'
    return ORIGINAL_COMPONENT_AUDIT(fit,expert)


def root_qualification(name):
    import torch
    torch.set_num_threads(2)
    from scripts.stvg_opd_paper_later_common_v1 import committed
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.decota_spatial_opd_chart_revision003 import chart_audit
    dest=BASE/'component_qualification'/name;b=read(dest/'PREDICTION_BARRIER.json')
    assert b['qualification'] and b['status']=='sealed' and not b['GT_read']
    assert read(REC/'qualification'/name/'BITWISE_CONTROLS.json')['status']=='pass'
    chains={};count=coordinates=0
    for key,rc in b['logical_records'].items():
        p=BASE/rc['path'];receipt=read(p.with_suffix('.json'))
        assert sha(p)==rc['sha256']==receipt['sha256'] and p.stat().st_size==receipt['bytes']
        z=load(p);fit=z['fit'];inp=load(BASE/z['input']['path'])
        assert sha(BASE/z['input']['path'])==z['input']['sha256']
        expert=unpack_expert(inp['expert']);checked=dispatch_saved(fit,expert,z)
        assert checked==z['math_audit'] and not z['GT_read']
        chain=(z['condition'],z['order'],z['arm']);prior=chains.get(chain)
        assert z['previous_payload_sha256']==(None if prior is None else prior[0])
        if prior is not None:
            assert all(torch.equal(v,torch.zeros_like(v) if n=='spatial.query_residual' else prior[1][n])
                       for n,v in fit['initial'].items())
        assert torch.count_nonzero(fit['initial']['spatial.query_residual'])==0
        expected=committed(fit['initial'],fit['state'],z['config']['writeback'])
        assert all(torch.equal(expected[n],z['committed'][n]) for n in expected)
        assert fit['selected_step']==z['config']['steps'] and fit['active_parameters']==1792
        if not fit.get('direct_objective'):assert chart_audit(fit)['status']=='pass'
        chains[chain]=(sha(p),z['committed']);count+=1
        coordinates+=sum(v.numel() for v in fit['state'].values())
    assert count==b['actual_fit_arrivals']==b['adapted_arrivals']
    path=REC/'qualification'/name/'ROOT_READBACK.json'
    if not path.exists():write(path,dict(status='pass',scope='actual saved complete P2 qualification fit/audit/state/chart readback',
        actual_fits=count,state_coordinates=coordinates,qualification_barrier_sha256=sha(dest/'PREDICTION_BARRIER.json'),
        bitwise_controls_sha256=sha(REC/'qualification'/name/'BITWISE_CONTROLS.json'),
        GT_read=False,new_model_calls=0,formal_predictions_accepted=0,time=time.time()))
    print('P2_ACTUAL_ROOT_QUALIFICATION_PASS',name,count,flush=True)


ORIGINAL_COMPONENT_AUDIT=None
def run(job,name,qualification=False):
    global ORIGINAL_COMPONENT_AUDIT
    verify_revision();assert name=='P2' or name.startswith('P2_')
    import vg_tta.stvg_opd_paper_component_audit_v1 as target
    ORIGINAL_COMPONENT_AUDIT=target.audit
    if job=='root_qualification':return root_qualification(name)
    if job=='score':
        def saved(fit,expert):
            f=sys._getframe(1)
            assert f.f_code.co_filename.endswith('score_stvg_opd_later_phase_v1.py')
            return dispatch_saved(fit,expert,f.f_locals['z'],f.f_locals['binding'])
        target.audit=saved
    elif job=='GPU':
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')==ALLOCATOR
        from vg_tta.decota_spatial_opd_chart_revision003 import Installation
        installation=Installation(qualification_vjp=qualification)
        comparisons=[];active=False
        def audited(fit,expert):
            nonlocal active
            frame=sys._getframe(1);loc=frame.f_locals
            assert frame.f_code.co_filename.endswith('run_stvg_opd_paper_components_v2.py')
            meta=dict(stage=name,arm=loc['arm'],condition=loc['condition'],order=loc['order'],
                      arrival=loc['at'],query_ordinal=loc['q'],GT_read=False)
            try:
                checked=component_audit(fit,expert)
                if qualification and not active:
                    active=True
                    import torch
                    # Ordinary complete old/new controls compare actual serialized
                    # tensors, never memory of a process which already exited.
                    torch.cuda.synchronize()
                    vjp=installation.last_vjp_checks if not fit.get('direct_objective') else []
                    with installation.original():
                        repeated=loc['fit_variant'](loc['base'],loc['initial'],expert,loc['row']['frame_ids'],
                            loc['row']['key'],loc['arm'],loc['cfg'])
                    torch.cuda.synchronize()
                    lhs={k:v for k,v in fit.items() if k!='numerical_chart'}
                    counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                    equal(lhs,repeated,counts)
                    comparisons.append(dict(**meta,**counts,actual_GPU_fits=2,original_and_bridge_complete_fit_bitwise=True,
                        native_head_VJP_checks=vjp,independent_full_decoder_Jacobian=False))
                    del repeated;active=False
                return checked
            except BaseException:
                folder=REC/'failures'/str(time.time_ns());folder.mkdir(parents=True,exist_ok=True)
                from methods.decota_final_simplified_v1.tensors import detached
                save(folder/'FAILED_FIT.pt',dict(fit=fit,expert=detached(expert,'cpu'),**meta))
                (folder/'traceback.txt').write_text(traceback.format_exc())
                write(folder/'CAPTURE_RECEIPT.json',dict(status='actual_complete_failed_fit_preserved_before_exit',**meta,
                    failed_fit_sha256=sha(folder/'FAILED_FIT.pt'),formal_prediction_accepted=False,time=time.time()))
                raise
        target.audit=audited
    else:assert job=='finalize'
    from scripts.run_stvg_opd_revision_later_v2 import run as original
    try:
        original(job,name,*(['qualification'] if qualification else []))
        if job=='GPU' and qualification:
            q=read(BASE/'component_qualification'/name/'QUALIFICATION.json')
            assert len(comparisons)==q['actual_fits']
            write(REC/'qualification'/name/'BITWISE_CONTROLS.json',dict(status='pass',scope='ordinary matched full old/new P2 fits',
                records=comparisons,qualification_arrivals=len(comparisons),actual_GPU_fits=2*len(comparisons),
                formal_predictions_accepted=0,GT_read=False,runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
    finally:
        target.audit=ORIGINAL_COMPONENT_AUDIT
        if job=='GPU':installation.uninstall()
        gc.collect()


if __name__=='__main__':
    try:run(sys.argv[1],sys.argv[2],len(sys.argv)>3 and sys.argv[3]=='qualification')
    except BaseException:
        p=REC/'worker_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc());raise
