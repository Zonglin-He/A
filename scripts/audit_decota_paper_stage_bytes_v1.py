"""Independent byte/receipt readback of sealed Table1 Ours and CPU controls.

This reads opaque prediction bytes and JSON receipts, never prediction arrays,
annotations, models, or GT metrics. It does not certify state/math/dense audits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/decota_paper_experiments_v1'
JOBS = [('t1_ours_hc2', 'hc2', 'vidstg', 3482, 3482, 237),
        ('t1_ours_vid', 'vidstg', 'hcstvg2', 10303, 732, 732)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_payload(base, relative, expected_hash, runtime_hash, seal_time):
    """Check opaque payload integrity and its independent sidecar receipt."""
    base = Path(base).resolve()
    p = base / relative
    assert not Path(relative).is_absolute() and p.resolve().is_relative_to(base)
    assert p.suffix == '.npz' and p.is_file()
    rc = read(p.with_suffix('.json'))
    assert rc['GT_read'] is False
    assert rc['runtime_lock_sha256'] == runtime_hash
    assert rc['sha256'] == expected_hash == sha(p)
    assert type(rc['bytes']) is int and rc['bytes'] == p.stat().st_size
    assert type(rc['time']) in (int, float) and math.isfinite(rc['time'])
    assert rc['time'] <= seal_time
    return rc['bytes']


def check_barrier(base, relative, expected_paths, runtime_hash):
    p = Path(base) / relative
    b = read(p)
    assert b['status'] == 'sealed' and b['GT_read'] is False
    assert type(b['time']) in (int, float) and math.isfinite(b['time'])
    assert set(b['files']) == set(expected_paths), 'Missing/extra sealed payload'
    assert len(b['files']) == len(expected_paths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        totals = list(pool.map(
            lambda x: check_payload(base, x[0], x[1], runtime_hash, b['time']),
            b['files'].items()))
    return b, dict(prediction_files_verified=len(totals),
                   prediction_bytes_verified=sum(totals),
                   prediction_barrier_sha256=sha(p),
                   receipt_GT_runtime_size_time_checks='pass')


def audit(base=BASE):
    base = Path(base)
    runtime_hash = sha(base / 'RUNTIME_LOCK.json')
    global_path = base / 'GLOBAL_PREDICTION_BARRIER.json'
    gb = read(global_path)
    assert gb['status'] == 'sealed' and gb['GT_read'] is False
    assert gb['cells'] == 41355 and set(gb['jobs']) == {x[0] for x in JOBS}
    ctrl_global_path = base / 'table1_stateless/GLOBAL_PREDICTION_BARRIER.json'
    cb = read(ctrl_global_path)
    assert cb['status'] == 'sealed' and cb['GT_read'] is False
    assert cb['unique_readouts_per_arm'] == 13785
    assert cb['logical_rows_per_arm'] == 41355
    assert cb['new_DINO_calls'] == cb['new_model_forwards'] == 0
    assert set(cb['jobs']) == {x[0] for x in JOBS}
    results = []
    for job, ds, source, queries, videos, parents in JOBS:
        paths = [f'{job}/online/clean/order{o}/{i:05}.npz'
                 for o in (1, 2, 3) for i in range(queries)]
        b, r = check_barrier(base, f'{job}/PREDICTION_BARRIER.json', paths, runtime_hash)
        assert b['job'] == job and b['dataset'] == ds and b['source_checkpoint'] == source
        assert b['cells'] == queries * 3
        assert r['prediction_barrier_sha256'] == gb['jobs'][job]
        assert b['time'] <= gb['time']
        control_paths = [f'table1_stateless/{job}/{i:05}.npz' for i in range(queries)]
        c, cr = check_barrier(base, f'table1_stateless/{job}/PREDICTION_BARRIER.json',
                              control_paths, runtime_hash)
        assert c['arms'] == ['Source Only', 'DINO-Refine']
        assert c['unique_readouts'] == queries and c['logical_rows_per_arm'] == queries * 3
        assert cr['prediction_barrier_sha256'] == cb['jobs'][job]
        assert c['time'] <= cb['time']
        results.append(dict(job=job, dataset=ds, source_checkpoint=source,
                            target_split='validation' if ds == 'hc2' else 'test',
                            queries=queries, official_videos_or_clips=videos,
                            parent_sources=parents, orders=3, arrivals=queries * 3,
                            Ours=r, CPU_stateless_controls=cr))
    return dict(status='passed_stage_seals_metadata_and_opaque_byte_hashes_only',
                jobs=results, Ours_arrivals_verified=41355,
                stateless_unique_payloads_verified=13785,
                stateless_logical_rows_per_arm=41355,
                runtime_lock_sha256=runtime_hash,
                Ours_global_barrier_sha256=sha(global_path),
                stateless_global_barrier_sha256=sha(ctrl_global_path),
                GT_read=False, prediction_arrays_read=False, efficacy_scoring=False,
                state_or_math_or_dense_audit=False, all_Table1_arms_complete=False,
                whole_paper_complete=False,
                next='Existing serial continuation: references, qualified ports/source Fisher; '
                     'GT scoring and pipeline diagnosis only after all Table1 deployable arms seal',
                time=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=BASE / 'TABLE1_OURS_CONTROLS_ROOT_BYTE_READBACK.json')
    args = parser.parse_args()
    result = audit()
    if args.output.exists():
        prior = read(args.output)
        assert {k:v for k,v in prior.items() if k != 'time'} == {
            k:v for k,v in result.items() if k != 'time'}
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
    print(json.dumps({k:result[k] for k in ['status', 'Ours_arrivals_verified',
                                         'stateless_unique_payloads_verified', 'GT_read']}))


if __name__ == '__main__':
    main()
