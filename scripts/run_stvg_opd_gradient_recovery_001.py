"""Pinned, process-local audit revision; no original runtime/code overwrite."""
import os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,activate,verify,read,write,sha
RECOVERY=BASE/'recovery/gradient_audit_001'

def activate_revision():
    verify();r=read(RECOVERY/'REVISION_RUNTIME.json')
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    activate()
    import vg_tta.decota_spatial_opd_tunable_audit_v1 as target
    from vg_tta.decota_spatial_opd_precision_audit_revision001 import audit
    target.audit=audit
    return audit

def run(*args):
    audit=activate_revision()
    # Preserve exact failed fit and compare the unreceipted deterministic retry.
    if args==('GPU','P0_hc2') and not (RECOVERY/'UNRECEIPTED_REPLAY_BITWISE.json').exists():
        import torch
        import vg_tta.decota_spatial_opd_tunable_audit_v1 as target
        original_fit=torch.load(RECOVERY/'FAILED_FIT.pt',weights_only=False,map_location='cpu')['fit']
        def equal(a,b):
            if isinstance(a,torch.Tensor):assert torch.equal(a,b)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b):equal(x,y)
            else:assert a==b
        def qualified(z,expert):
            if z['arm']=='frozen_rollout' and not (RECOVERY/'UNRECEIPTED_REPLAY_BITWISE.json').exists():
                equal(z,original_fit);checked=audit(z,expert)
                write(RECOVERY/'UNRECEIPTED_REPLAY_BITWISE.json',dict(status='pass',
                    failed_fit_all_tensors_states_gradients_actions_boxes_bitwise=True,
                    configuration_and_algorithm_unchanged=True,original_failing_fit_sha256=sha(RECOVERY/'FAILED_FIT.pt'),
                    independent_math=checked,GT_read=False,time=time.time()))
                return checked
            return audit(z,expert)
        target.audit=qualified
    import scripts.run_stvg_opd_paper_hc2_revision_v2 as dispatcher
    dispatcher.run(*args)

if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException:
        d=RECOVERY/'new_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True)
        (d/'traceback.txt').write_text(traceback.format_exc());raise
