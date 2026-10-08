"""Receipt-aware process-local precision audit; unchanged P1 fitter and state.

GPU_requal repeats the reproduced failed fit, checks every tensor, then stops
before writing a prediction. GPU continues only after that actual pass.
Postseal CPU readback dispatches by the saved audit revision without rewriting
the old prefix or changing its immutable math-audit receipts.
"""
import os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha,verify,activate
from scripts.decota_matrix_common_v1 import save,load
REC=BASE/'recovery/P1_softmax_precision_002'

class RequalificationComplete(Exception):pass

def verify_revision():
    verify();rt=read(REC/'REVISION_RUNTIME.json')
    for f,h in rt['pins'].items():assert sha(ROOT/f)==h,f
    for f,item in read(REC/'CAPTURE_RECEIPT.json')['original_code'].items():
        assert sha(ROOT/f)==item['sha256'],f
    assert sha(BASE/'RUNTIME_LOCK.json')==rt['original_runtime_sha256']
    assert sha(BASE/'P1_PRECISION_RUNTIME.json')==rt['previous_precision_runtime_sha256']
    import torch
    assert torch.version.git_version==rt['torch_git_version']
    return rt

def equal(a,b,counts):
    import torch
    import numpy as np
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
        counts['tensors']+=1;counts['tensor_coordinates']+=a.numel()
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k],counts)
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y,counts)
    else:assert a==b;counts['scalar_values']+=1

def caller_metadata(frame):
    loc=frame.f_locals
    return dict(stage=loc['stage_name'],order=loc['order'],arrival=loc['at'],
                query_ordinal=loc['q'],input=loc['inputrc'],
                previous_payload_sha256=loc['prevhash'][loc['arm']],
                source=loc['stage']['source'],GT_read=False)

def run(job,*args):
    verify_revision();activate()
    from scripts.run_stvg_opd_gradient_recovery_001 import activate_revision
    previous_audit=activate_revision()
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import audit as current_audit,original_audit
    import vg_tta.decota_spatial_opd_tunable_audit_v1 as target
    from methods.decota_final_simplified_v1.tensors import detached
    assert read(REC/'CPU_REQUALIFICATION.json')['status']=='pass'
    requal=job=='GPU_requal'
    if job in ('GPU','GPU_requal'):
        if not requal:assert read(REC/'GPU_REQUALIFICATION.json')['status']=='pass'
        expected=load(REC/'REPRODUCED_FAILED_FIT.pt') if args==('P1_hc2',) else None
        def gpu_audit(fit,expert):
            frame=sys._getframe(1);meta=caller_metadata(frame)
            try:
                if expected is not None and meta['order']=='order2' and meta['arrival']==148:
                    assert meta['query_ordinal']==expected['query_ordinal']
                    assert meta['previous_payload_sha256']==expected['previous_payload_sha256']
                    assert meta['input']['sha256']==expected['input']['sha256']
                    counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                    equal(fit,expected['fit'],counts);equal(detached(expert,'cpu'),expected['expert'],counts)
                    checked=current_audit(fit,expert)
                    receipt=dict(status='pass',**meta,**counts,
                        repeated_fit_all_tensors_actions_weights_gradients_Adam_states_boxes_bitwise=True,
                        serialized_reproduction_sha256=sha(REC/'REPRODUCED_FAILED_FIT.pt'),
                        original_failed_fit_not_serialized=True,
                        equality_is_between_two_reproductions_from_saved_prefix=True,
                        new_predictions_saved=False if requal else 'next operation writes only previously unreceipted arrival148',
                        independent_math=checked,
                        fit_GPU_seconds=frame.f_locals['seconds'],
                        shared_capture_seconds=frame.f_locals['capture_seconds'],
                        source_config_and_original_science_unchanged=True,
                        engineering_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),time=time.time())
                    name='GPU_REQUALIFICATION.json' if requal else 'UNRECEIPTED_FORMAL_REPLAY_BITWISE.json'
                    if not (REC/name).exists():write(REC/name,receipt)
                    if requal:raise RequalificationComplete()
                    return checked
                assert not requal,'Requalification must stop at exact first unreceipted arrival'
                return current_audit(fit,expert)
            except RequalificationComplete:raise
            except BaseException:
                dest=REC/'new_failures'/str(time.time_ns());dest.mkdir(parents=True,exist_ok=True)
                save(dest/'FAILED_FIT.pt',dict(fit=fit,expert=detached(expert,'cpu'),**meta))
                (dest/'traceback.txt').write_text(traceback.format_exc())
                write(dest/'FAILURE_RECEIPT.json',dict(status='actual_new_failure_preserved',**meta,
                    failed_fit_sha256=sha(dest/'FAILED_FIT.pt'),time=time.time()))
                raise
        target.audit=gpu_audit
    elif job=='P1_score':
        assert read(BASE/'P1_PREDICTION_BARRIER.json')['all_deployment_OPD_directions']
        def receipt_audit(fit,expert):
            frame=sys._getframe(1)
            assert frame.f_code.co_filename.endswith('score_stvg_opd_p1_v1.py')
            saved=frame.f_locals['z']['math_audit'];revision=saved.get('revision')
            if revision is None:return original_audit(fit,expert)
            if revision=='precision_audit_revision001':return previous_audit(fit,expert)
            assert revision=='precision_audit_revision002',revision
            return current_audit(fit,expert)
        target.audit=receipt_audit
    else:target.audit=current_audit
    from scripts.run_stvg_opd_paper_hc2_revision_v2 import run as dispatch
    try:dispatch('GPU' if requal else job,*args)
    except RequalificationComplete:
        print('ACTUAL_P1_FAILURE_REPEAT_BITWISE_AND_REVISION002_REQUALIFICATION_PASS_NO_PREDICTION_SAVED',flush=True)

if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException:
        dest=REC/'worker_failures'/str(time.time_ns());dest.mkdir(parents=True,exist_ok=True)
        (dest/'traceback.txt').write_text(traceback.format_exc());raise
