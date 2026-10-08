"""Reproduce the first unreceipted P1 fit under the original audit, then stop.

The old worker did not serialize its failed fit. This capture is a deterministic
reproduction from the exact complete saved prefix, not bytes from dead memory.
No prediction is accepted, GT is unopened, and algorithm/runtime bytes are unchanged.
"""
import os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha,verify
from scripts.decota_matrix_common_v1 import save
from scripts.run_stvg_opd_gradient_recovery_001 import activate_revision
REC=BASE/'recovery/P1_softmax_precision_002'

class CapturedOriginalAuditFailure(Exception): pass

def main():
    verify()
    assert read(REC/'CAPTURE_RECEIPT.json')['done']==3630
    runtime=read(REC/'REPRODUCTION_RUNTIME_IMPORT_REVISION.json')
    for f,h in runtime['pins'].items():assert sha(ROOT/f)==h,f
    audit=activate_revision()
    import vg_tta.decota_spatial_opd_tunable_audit_v1 as target
    from methods.decota_final_simplified_v1.tensors import detached
    def capture(z,expert):
        frame=sys._getframe(1)
        assert frame.f_code.co_filename.endswith('run_stvg_opd_paper_v1.py')
        local=frame.f_locals
        assert local['stage_name']=='P1_hc2' and local['order']=='order2' and local['at']==148
        record=dict(fit=z,expert=detached(expert,'cpu'),stage=local['stage_name'],
            order=local['order'],arrival=local['at'],query_ordinal=local['q'],
            input=local['inputrc'],previous_payload_sha256=local['prevhash'][local['arm']],
            interval=local['native']['physical_interval'],row_key=local['row']['key'],
            source=local['stage']['source'],GT_read=False,
            replay_from_exact_saved_prefix=True,original_failed_fit_not_serialized=True)
        path=REC/'REPRODUCED_FAILED_FIT.pt'
        save(path,record)
        try:
            audit(z,expert)
        except AssertionError:
            text=traceback.format_exc()
            assert 'abs(ew-w).max()<3e-7' in text
            (REC/'ORIGINAL_AUDIT_REPRODUCTION_TRACEBACK.txt').write_text(text)
            write(REC/'REPRODUCTION_RECEIPT.json',dict(status='original_softmax_failure_reproduced',
                failed_fit_sha256=sha(path),previous_payload_sha256=record['previous_payload_sha256'],
                input=record['input'],stage=record['stage'],order=record['order'],arrival=record['arrival'],
                query_ordinal=record['query_ordinal'],GT_read=False,new_prediction_saved=False,
                original_failed_fit_was_not_serialized=True,
                capture_is_deterministic_reproduction_not_dead_memory=True,
                runtime_sha256=sha(REC/'REPRODUCTION_RUNTIME_IMPORT_REVISION.json'),time=time.time()))
            raise CapturedOriginalAuditFailure()
        raise AssertionError('Original softmax failure did not reproduce; preserve capture and diagnose.')
    target.audit=capture
    from scripts.run_stvg_opd_paper_hc2_revision_v2 import run
    try:run('GPU','P1_hc2')
    except CapturedOriginalAuditFailure:
        print('ORIGINAL_P1_SOFTMAX_FAILURE_REPRODUCED_NO_PREDICTION_SAVED',flush=True)

if __name__=='__main__':main()
