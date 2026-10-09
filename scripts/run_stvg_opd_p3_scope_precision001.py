"""Original locked P3 with separately pinned parameter-scope numeric audit."""
import gc
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,read,write,status,sha
activate()
from scripts.decota_matrix_common_v1 import load,save
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
from scripts.run_stvg_opd_p2_matched_gradient006 import verify as prior_verify,dispatch_saved as prior_dispatch,prior
from vg_tta.stvg_opd_p3_scope_precision001 import Installation,audit,REVISION,FIELD,ACTION
REC=BASE/'P3_scope_engineering_revision001';ALLOCATOR='expandable_segments:True'


def verify():
    prior_verify();r=read(REC/'REVISION_RUNTIME.json')
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    assert sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json')==r['original_component_runtime_sha256']
    assert sha(BASE/'LATER_DESIGN_LOCK.json')==r['original_later_design_sha256']
    assert sha(REC/'CPU_CONTRACTS_revision002.json')==r['CPU_contracts_sha256']
    assert read(BASE/'P2_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    assert read(BASE/'P2_FINAL_GITHUB_RECEIPT.json')['status']=='pass'
    return r


def component_audit(fit,expert):
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit as original
    # The process hook is restored to the captured immutable audit before use.
    return dict(audit(fit,expert,ORIGINAL_AUDIT),revision=REVISION,
        parameter_scope_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),
        original_component_runtime_sha256=sha(BASE/'COMPONENT_RUNTIME_LOCK_revision001.json'))


def dispatch_saved(fit,expert,z,binding=None):
    if z['math_audit'].get('revision')==REVISION:
        assert z['scope_precision_runtime_sha256']==sha(REC/'REVISION_RUNTIME.json')
        assert fit[FIELD]['revision']==REVISION and z['stage'].startswith('P3_')
        return component_audit(fit,expert)
    assert fit.get(FIELD,{}).get('revision')!=REVISION,'Unknown scoped audit receipt'
    prior.previous.ORIGINAL_COMPONENT_AUDIT=ORIGINAL_AUDIT
    return prior_dispatch(fit,expert,z,binding)


def scientific(fit):return {k:v for k,v in fit.items() if k not in [FIELD,ACTION,'numerical_chart']}


def gpu_run(name,qualify):
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.stvg_opd_legacy_input_bridge003 import install
    import scripts.run_stvg_opd_paper_components_v2 as runner
    import vg_tta.stvg_opd_paper_component_audit_v1 as component
    from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
    install();original_save=runner.save;old_audit=component.audit;component.audit=component_audit
    recorder=Installation(qualification=qualify);checks=[]
    def checked_save(path,z):
        if not isinstance(z,dict) or 'fit' not in z:return original_save(path,z)
        assert z['stage']==name and z['stage'].startswith('P3_') and not z['GT_read']
        z['scope_precision_runtime_sha256']=sha(REC/'REVISION_RUNTIME.json')
        if qualify:
            frame=sys._getframe(1)
            while frame and not frame.f_code.co_filename.endswith('run_stvg_opd_paper_components_v2.py'):frame=frame.f_back
            assert frame is not None;loc=frame.f_locals
            assert all(loc[k] is not None for k in ['base','initial','ex','row','model','expert'])
            vjp=list(recorder.last_vjp)
            with recorder.original():
                raw=fit_variant(loc['base'],loc['initial'],loc['ex'],loc['row']['frame_ids'],loc['row']['key'],z['arm'],z['config'])
            torch.cuda.synchronize();counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            equal(scientific(z['fit']),raw,counts)
            assert state_hash(loc['model'].state_dict())==loc['modelhash']
            assert state_hash(loc['expert'].model.state_dict())==loc['experthash']
            control=REC/'qualification'/name/z['condition']/z['order']/z['arm']/(str(z['arrival']).zfill(5)+'.pt')
            original_save(control,raw)
            checks.append(dict(stage=name,condition=z['condition'],order=z['order'],arm=z['arm'],arrival=z['arrival'],
                query_ordinal=z['query_ordinal'],active_parameters=z['fit']['active_parameters'],
                real_original_and_new_complete_fits=2,original_new_scientific_result_bitwise=counts,
                qualified_path=str(path.relative_to(BASE)),original_fit_path=str(control.relative_to(BASE)),
                original_fit_sha256=sha(control),input_sha256=z['input']['sha256'],
                native_head_VJP=vjp,source_and_expert_process_hashes_unchanged=True,
                formal_predictions_accepted=0,GT_read=False))
        elif z['arrival']<2 and z['order']=='order1':
            f=BASE/'component_qualification'/name/z['condition']/z['order']/z['arm']/path.name
            assert f.exists();qualified=load(f);counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            assert qualified['input']['sha256']==z['input']['sha256'] and qualified['config']==z['config']
            equal(scientific(z['fit']),scientific(qualified['fit']),counts)
            receipt=REC/'first_formal'/name/z['condition']/z['order']/z['arm']/(path.stem+'.json')
            write(receipt,dict(status='pass',scope='first two original arrivals require qualified fit equality before acceptance',
                stage=name,condition=z['condition'],order=z['order'],arm=z['arm'],arrival=z['arrival'],query_ordinal=z['query_ordinal'],
                complete_fit_bitwise=counts,qualified_fit_sha256=sha(f),GT_read=False,time=time.time()))
        return original_save(path,z)
    runner.save=checked_save
    try:
        runner.run(name,qualify)
        if qualify:
            barrier=read(BASE/'component_qualification'/name/'PREDICTION_BARRIER.json')
            assert len(checks)==barrier['adapted_arrivals']
            for c in checks:c['qualified_fit_sha256']=sha(BASE/c['qualified_path'])
            write(REC/'qualification'/name/'BITWISE_CONTROLS.json',dict(status='pass',scope='actual complete original/new fits for all original P3 qualification cells',
                records=checks,qualification_arrivals=len(checks),actual_GPU_fits=2*len(checks),GT_read=False,
                original_source_runtime_and_science_unchanged=True,independent_full_decoder_Jacobian=False,time=time.time()))
    finally:
        runner.save=original_save;component.audit=old_audit;recorder.uninstall();gc.collect()


def root_qualification():
    import torch
    from scripts.stvg_opd_paper_later_common_v1 import phases,stage_definition,committed
    from scripts.run_decota_paper_main_v1 import unpack_expert
    counts=dict(complete_fit_math_exact=0,original_new_bitwise_pairs=0,rounds=0,state_coordinates=0);records=[]
    for name in phases()['P3']:
        stage=stage_definition(name);barrier=read(BASE/'component_qualification'/name/'PREDICTION_BARRIER.json')
        control=read(REC/'qualification'/name/'BITWISE_CONTROLS.json')
        assert control['status']=='pass' and control['qualification_arrivals']==barrier['adapted_arrivals']
        for c in control['records']:
            f=BASE/c['qualified_path'];assert sha(f)==c['qualified_fit_sha256'];z=load(f);fit=z['fit']
            assert z['config']==stage['variant_configs'][z['arm']] and z['GT_read'] is False
            assert fit['active_parameters']=={'query_only':256,'LN_only':1536}.get(z['arm'],1792)
            inp=BASE/z['input']['path'];assert sha(inp)==z['input']['sha256'];data=load(inp);ex=unpack_expert(data['expert'])
            assert data['GT_read'] is False and z['interval']==data['interval']
            assert component_audit(fit,ex)==z['math_audit']
            expected=committed(fit['initial'],fit['state'],z['config']['writeback'])
            assert all(torch.equal(expected[k],z['committed'][k]) for k in expected)
            assert torch.count_nonzero(fit['initial']['spatial.query_residual'])==0
            if z['arrival']>0:
                prev=load(f.with_name('00000.pt'));assert z['previous_payload_sha256']==sha(f.with_name('00000.pt'))
                assert all(torch.equal(v,torch.zeros_like(v) if k=='spatial.query_residual' else prev['committed'][k]) for k,v in fit['initial'].items())
            rawfile=BASE/c['original_fit_path'];assert sha(rawfile)==c['original_fit_sha256']
            compare=dict(tensors=0,tensor_coordinates=0,scalar_values=0);equal(scientific(fit),load(rawfile),compare);assert compare==c['original_new_scientific_result_bitwise']
            assert len(c['native_head_VJP'])==(fit['config']['steps'] if fit['positions'] else 0)
            counts['complete_fit_math_exact']+=1;counts['original_new_bitwise_pairs']+=1;counts['rounds']+=len(fit['rounds']);counts['state_coordinates']+=1792
            records.append({k:c[k] for k in ['stage','condition','order','arm','arrival','query_ordinal','active_parameters','qualified_fit_sha256','original_fit_sha256']})
    assert counts['complete_fit_math_exact']==96
    receipt=dict(status='pass',scope='actual P3 complete qualified input/math/scoped optimizer/inactive state/reset/LN-writeback and saved complete original-new controls',
        counts=counts,qualification_arrivals=96,actual_GPU_fits=192,records=records,GT_read=False,
        formal_predictions_accepted=0,independent_full_decoder_Jacobian=False,time=time.time())
    write(REC/'ROOT_QUALIFICATION_READBACK.json',receipt)
    write(BASE/'P3_QUALIFICATION.json',dict(receipt,runtime_sha256=sha(REC/'REVISION_RUNTIME.json')))


def run(job,name):
    verify()
    if job=='qualification':return gpu_run(name,True)
    if job=='GPU':return gpu_run(name,False)
    if job=='root_qualification':return root_qualification()
    import vg_tta.stvg_opd_paper_component_audit_v1 as component
    old=component.audit
    if job=='finalize':
        from scripts.finalize_stvg_opd_later_phase_v1 import run as finalize
        return finalize(name)
    assert job=='score' and name=='P3'
    from scripts.stvg_opd_legacy_input_bridge003 import install
    install()
    from scripts.score_stvg_opd_later_phase_v1 import run as scoring
    def saved_audit(fit,expert):
        frame=sys._getframe(1)
        assert frame.f_code.co_filename.endswith('score_stvg_opd_later_phase_v1.py')
        return dispatch_saved(fit,expert,frame.f_locals['z'],frame.f_locals['binding'])
    component.audit=saved_audit
    try:return scoring(name)
    finally:component.audit=old


from vg_tta.stvg_opd_paper_component_audit_v1 import audit as ORIGINAL_AUDIT


if __name__=='__main__':
    try:run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'P3')
    except BaseException:
        p=REC/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'COMPONENT_STAGE.json',dict(status='failed_preserved',phase='P3',evidence=str(p.relative_to(ROOT)),GT_read=False,time=time.time()));raise
