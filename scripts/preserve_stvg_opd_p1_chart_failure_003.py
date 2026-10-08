"""Preserve the actual P1 VidSTG chart failure and opaque receipted prefix."""
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_opd_paper_hc2_revision_v2'
REC = BASE / 'recovery/P1_box_chart_003'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not (REC / 'CAPTURE_RECEIPT.json').exists()
    REC.mkdir(parents=True, exist_ok=True)
    status = read(BASE / 'STATUS.json')
    assert status['status'] == 'new_revision002_failure_preserved'
    design = read(BASE / 'DESIGN_LOCK.json')
    stage = design['stages']['P1_vidstg']
    assert stage['orders']['order1'][1882] == 10220
    runtime = sha(BASE / 'RUNTIME_LOCK.json')
    files = {}
    for at in range(1882):
        path = BASE / 'stages/P1_vidstg/clean/order1/on_policy' / f'{at:05}.pt'
        receipt = read(path.with_suffix('.json'))
        digest = sha(path)
        assert receipt['sha256'] == digest and receipt['bytes'] == path.stat().st_size
        assert receipt['GT_read'] is False and receipt['runtime_lock_sha256'] == runtime
        files[str(path.relative_to(ROOT))] = dict(sha256=digest, bytes=path.stat().st_size,
            receipt_sha256=sha(path.with_suffix('.json')))
    assert not (BASE / 'stages/P1_vidstg/clean/order1/on_policy/01882.pt').exists()
    assert not (BASE / 'stages/P1_vidstg/PREDICTION_BARRIER.json').exists()
    assert not (BASE / 'P1_PREDICTION_BARRIER.json').exists()
    paths = ['scripts/run_stvg_opd_p1_softmax_recovery_002.py',
        'scripts/continue_stvg_opd_p1_softmax_recovery_002.py',
        'vg_tta/decota_spatial_opd_tunable_v1.py', 'vg_tta/c1_enabling_tricks_v1.py',
        'vg_tta/decota_spatial_opd_precision_audit_revision002.py',
        'scripts/run_stvg_opd_paper_v1.py',
        'external/TA-STVG/models/grounding_model/query_decoder.py',
        'methods/decota_spatial_opd_v1/configs.json']
    paths += [str((BASE / rel).relative_to(ROOT)) for rel in [
        'RUNTIME_LOCK.json', 'DESIGN_LOCK.json', 'STAGE.json', 'STATUS.json',
        'stages/P1_vidstg/STATUS.json', 'P1_PRECISION_RUNTIME.json',
        'recovery/P1_softmax_precision_002/REVISION_RUNTIME.json',
        'P1_HC2_ROOT_BYTE_READBACK.json', 'P1_HC2_ROOT_BYTE_READBACK_SNAPSHOT_VERIFIED.json',
        'recovery/P1_softmax_precision_002/continuation_GPU_P1_vidstg.log']]
    for rel in ['worker_failures/1791472289632956870',
                'controller_failures/1791472290395373832']:
        paths += [str(p.relative_to(ROOT)) for p in
            (BASE / 'recovery/P1_softmax_precision_002' / rel).rglob('*') if p.is_file()]
    originals = {}
    for rel in paths:
        source = ROOT / rel
        copy = REC / 'originals' / rel
        copy.parent.mkdir(parents=True, exist_ok=True)
        if copy.exists():
            assert sha(copy) == sha(source)
        else:
            shutil.copy2(source, copy)
        assert sha(copy) == sha(source)
        originals[rel] = dict(sha256=sha(source), bytes=source.stat().st_size,
            preserved_path=str(copy.relative_to(ROOT)))
    inp = BASE / 'inputs' / f'{stage["source"]}_to_{stage["dataset"]}' / 'clean/10220.pt'
    rc = read(inp.with_suffix('.json'))
    assert rc['sha256'] == sha(inp) and rc['GT_read'] is False
    result = dict(status='actual_new_policy_chart_failure_preserved', stage='P1_vidstg',
        source=stage['source'], order='order1', done=1882, total=30909,
        failed_arrival=1882, query_ordinal=10220, original_code=originals,
        prefix_files=files, prefix_bytes=sum(v['bytes'] for v in files.values()),
        failed_input=dict(path=str(inp.relative_to(ROOT)), sha256=sha(inp),
            receipt_sha256=sha(inp.with_suffix('.json'))),
        all_original_prediction_receipts_verified=True, original_failed_fit_not_serialized=True,
        GT_read=False, global_P1_barrier_present=False, new_prediction_saved=False,
        HC2_adapted_sealed=10446, time=time.time())
    with (REC / 'CAPTURE_RECEIPT.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'done', 'total', 'prefix_bytes', 'GT_read']}))


if __name__ == '__main__':
    main()
