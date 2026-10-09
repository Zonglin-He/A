"""Bounded strict input-schema recovery; P2 fitting and audit002 stay intact."""
import gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,activate,read,write,status,sha
from scripts.decota_matrix_common_v1 import load,save
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
import scripts.run_stvg_opd_p2_engineering002 as original
REC=BASE/'recovery/P2_input_schema_003'
ALLOCATOR=original.ALLOCATOR


class QualificationComplete(Exception):pass


def verify():
    original.verify_revision();r=read(REC/'REVISION_RUNTIME.json')
    assert sha(original.REC/'REVISION_RUNTIME.json')==r['previous_engineering_runtime_sha256']
    assert sha(REC/'CAPTURE_RECEIPT.json')==r['capture_sha256']
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    assert sha(REC/'CPU_ALL_LEGACY_ROOT_READBACK.json')==r['CPU_legacy_readback_sha256']
    assert read(REC/'CPU_ALL_LEGACY_ROOT_READBACK.json')['status']=='pass'
    assert read(BASE/'P2_QUALIFICATION.json')['actual_GPU_fits']==192
    return r


def root_qualification(name):
    import torch
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit
    from scripts.stvg_opd_paper_later_common_v1 import committed
    folder=REC/'qualification'/name;z=load(folder/'QUALIFIED_FIRST_FORMAL.pt');f=z['fit']
    inp=load(BASE/z['input']['path']);assert sha(BASE/z['input']['path'])==z['input']['sha256']
    assert audit(f,unpack_expert(inp['expert']))==z['math_audit']
    assert f['direct_objective'] and f['active_parameters']==1792 and f['selected_step']==z['config']['steps']
    assert torch.count_nonzero(f['initial']['spatial.query_residual'])==0 and z['previous_payload_sha256'] is None
    expected=committed(f['initial'],f['state'],z['config']['writeback'])
    assert all(torch.equal(expected[n],z['committed'][n]) for n in expected)
    q=read(folder/'GPU_QUALIFICATION.json');assert q['status']=='pass' and q['actual_GPU_fits']==2
    write(folder/'ROOT_READBACK.json',dict(status='pass',scope='actual first formal input compatibility / complete direct fit / qualified-repeat state and original mathematical readback',
        stage=name,actual_GPU_fits=2,qualified_fit_sha256=sha(folder/'QUALIFIED_FIRST_FORMAL.pt'),
        GPU_qualification_sha256=sha(folder/'GPU_QUALIFICATION.json'),input_sha256=z['input']['sha256'],
        state_coordinates=1792,GT_read=False,formal_predictions_accepted=0,time=time.time()))


def run(job,name):
    verify();activate()
    from scripts.stvg_opd_legacy_input_bridge003 import install
    install()
    if job=='root_qualification':return root_qualification(name)
    if job not in ['GPU','GPU_requal']:
        assert job in ['score','finalize'] and name=='P2'
        return original.run(job,name)
    assert name.startswith('P2_') and os.environ.get('PYTORCH_CUDA_ALLOC_CONF')==ALLOCATOR
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    import scripts.run_stvg_opd_paper_components_v2 as runner
    oldsave=runner.save;seen=False;qualification=job=='GPU_requal'
    if not qualification:
        for s in ['P2_hc2_cross_clean','P2_vidstg_cross_clean']:
            assert read(REC/'qualification'/s/'ROOT_READBACK.json')['status']=='pass'
    def checked_save(path,z):
        nonlocal seen
        if not str(path).startswith(str(BASE/'stages')) or not isinstance(z,dict) or 'fit' not in z:
            return oldsave(path,z)
        assert z['stage']==name and z['GT_read'] is False
        first=z['condition']=='clean' and z['order']=='order1' and z['arrival']==0 and z['arm']=='direct_L1_GIoU'
        if first and name in ['P2_hc2_cross_clean','P2_vidstg_cross_clean']:
            seen=True;folder=REC/'qualification'/name
            expected=load(BASE/'component_qualification'/name/'clean/order1/direct_L1_GIoU/00000.pt')
            assert z['query_ordinal']==expected['query_ordinal'] and z['input']['sha256']==expected['input']['sha256']
            counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            equal(z['fit'],expected['fit'],counts);assert z['math_audit']==expected['math_audit']
            if qualification:
                assert not path.exists() and not folder.exists()
                frame=sys._getframe(1);loc=frame.f_locals
                assert frame.f_code.co_filename.endswith('run_stvg_opd_paper_components_v2.py')
                torch.cuda.synchronize()
                repeated=loc['fit_variant'](loc['base'],loc['initial'],loc['ex'],loc['row']['frame_ids'],
                    loc['row']['key'],loc['arm'],loc['cfg'])
                torch.cuda.synchronize();repeat=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                equal(z['fit'],repeated,repeat)
                assert original.ORIGINAL_COMPONENT_AUDIT(repeated,loc['ex'])==z['math_audit']
                assert state_hash(loc['model'].state_dict())==loc['modelhash']
                assert state_hash(loc['expert'].model.state_dict())==loc['experthash']
                folder.mkdir(parents=True)
                save(folder/'QUALIFIED_FIRST_FORMAL.pt',z)
                write(folder/'GPU_QUALIFICATION.json',dict(status='pass',scope='strict legacy input binding with unchanged full original direct fit',
                    stage=name,query_ordinal=z['query_ordinal'],actual_GPU_fits=2,
                    prior_real_qualified_fit_bitwise=counts,repeated_complete_fit_bitwise=repeat,
                    qualified_fit_sha256=sha(folder/'QUALIFIED_FIRST_FORMAL.pt'),input_sha256=z['input']['sha256'],
                    source_and_expert_state_unchanged=True,GT_read=False,new_DINO_calls=z['input']['new_DINO_calls'],
                    formal_predictions_accepted=0,runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time()))
                raise QualificationComplete()
            qualified=load(folder/'QUALIFIED_FIRST_FORMAL.pt');actual=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            equal(z['fit'],qualified['fit'],actual)
            if not (folder/'FIRST_FORMAL_FIT_BITWISE.json').exists():
                write(folder/'FIRST_FORMAL_FIT_BITWISE.json',dict(status='pass',stage=name,
                    query_ordinal=z['query_ordinal'],qualified_full_fit_bitwise=actual,
                    qualified_fit_sha256=sha(folder/'QUALIFIED_FIRST_FORMAL.pt'),GT_read=False,
                    next_operation='accept original first formal prediction after exact qualified comparison',time=time.time()))
        z['input_schema_runtime_sha256']=sha(REC/'REVISION_RUNTIME.json')
        return oldsave(path,z)
    runner.save=checked_save
    try:
        original.run('GPU',name)
        assert not qualification,'Qualification must stop before the first formal prediction save'
    except QualificationComplete:
        assert qualification and seen
        print('P2_STRICT_INPUT_SCHEMA_2_REAL_FITS_PASS_ZERO_FORMAL_ACCEPTED',name,flush=True)
    finally:runner.save=oldsave;gc.collect()


if __name__=='__main__':
    try:run(sys.argv[1],sys.argv[2])
    except BaseException:
        p=REC/'new_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True)
        (p/'traceback.txt').write_text(traceback.format_exc());raise
