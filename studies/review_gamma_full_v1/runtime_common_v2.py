"""Engineering-only complete Source checkpoint before equality or extraction."""
from gamma_full_common import *
def validate_runtime():
    p=read(D/'execution_v2_plan.json')
    assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    assert sha(D/'plan.json')==p['science_plan_sha256']
    for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
    assert read(D/'runtime_v2_static_proof.json')['worker_reverse_edits_exact_original']
    return p
def dispose_source_checkpoint(scratch):
    path=scratch/'Source_checkpoint.pt'
    if path.exists():
        meta=read(path.with_suffix('.json'))
        assert path.resolve().is_relative_to((D/'active').resolve())and sha(path)==meta['sha256']
        path.unlink();atomic(scratch/'Source_checkpoint_disposed.json',dict(sha256=meta['sha256'],after_committed_prediction_compact_receipt_cold_readback=True))
