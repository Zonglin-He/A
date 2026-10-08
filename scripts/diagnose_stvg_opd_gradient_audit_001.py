"""Reproduce the first unreceipted fit; preserve it before independent diagnosis.

No GT, output acceptance, prediction overwrite, scientific change, or retry loop.
"""
import json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,verify

def main():
    verify();activate()
    import torch
    import vg_tta.decota_spatial_opd_tunable_audit_v1 as module
    old=module.audit;dest=BASE/'recovery/gradient_audit_001'
    assert not (dest/'FAILED_FIT.pt').exists()
    def diagnose(z,expert):
        try:return old(z,expert)
        except AssertionError:
            torch.save(dict(fit=z,expert=expert,GT_read=False),dest/'FAILED_FIT.pt')
            (dest/'REPRODUCED_TRACEBACK.txt').write_text(traceback.format_exc())
            (dest/'REPRODUCTION.json').write_text(json.dumps(dict(
                status='same_original_independent_audit_assertion_reproduced',
                arm=z['arm'],config=z['config'],rounds=len(z['rounds']),
                gradient_calls=z['gradient_calls'],GT_read=False,
                accepted_prediction=False,original_760_predictions_unchanged=True,
                time=time.time()),indent=2)+'\n')
            raise
    module.audit=diagnose
    import scripts.run_stvg_opd_paper_v1 as worker
    worker.run('P0_hc2')

if __name__=='__main__':main()
