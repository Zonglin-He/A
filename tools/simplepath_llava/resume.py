"""Resume only frozen full stages after explicit user pause and TubeDETR completion."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

import numpy as np
from . import run as original

PLAN = original.PUBLIC / 'resume_plan.json'
PREFLIGHT = original.PUBLIC / 'resume_preflight.json'


def stages(plan):
    return [(stage, ds) for ds in plan['full_order'] for stage in ['gpu', 'score']]


def no_active_backbone():
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            args = [v.decode() for v in (path / 'cmdline').read_bytes().split(b'\0') if v]
            if '-m' in args:
                module = args[args.index('-m') + 1]
                assert module not in ['tools.simplepath_llava.run', 'tools.simplepath_llava.resume',
                                      'tools.tubedetr_spatial_search.priority_resume'], ('active backbone', path.name)
        except (OSError, UnicodeError, IndexError):
            continue


def preflight():
    """Validate every committed partial prediction before scheduling unchanged workers."""
    from tools.simplepath_tube.core.prototype import update, inclusive_scores
    p = original.checked()
    original.gate_both128(p)
    tube = json.loads((original.ROOT / 'outputs/maintenance/tubedetr_spatial_search_final_publication.json').read_text())
    assert tube['final_full_results_published'] and tube['all_remote_bytes_match']
    assert not (original.OUT / 'failure.json').exists()
    no_active_backbone()
    summaries = {}
    for ds in p['full_order']:
        rows = original.rows_for(p, ds, True)
        target = original.OUT / ds / 'full'
        assert not list(target.glob('*.pending')), 'Uncommitted temporary prediction retained for audit'
        payloads = {path.stem for path in target.glob('*.npz')}
        receipts = {path.stem for path in target.glob('*.json') if path.stem not in ['sealed', 'metrics', 'details_private']}
        assert payloads == receipts and payloads <= {row['id'] for row in rows}
        oldroot = original.ROOT / 'outputs/llava_st_development' / ds / 'full'
        oldseal = json.loads((oldroot / 'sealed.json').read_text())
        manifest = hashlib.sha256()
        count = 0; restored = 0; finite = 0; invalid = 0
        for row in rows:
            q = row['id']
            if q not in payloads:
                continue
            path = target / (q + '.npz'); receipt = path.with_suffix('.json')
            meta = json.loads(receipt.read_text()); h = original.sha(path)
            assert meta['payload_sha256'] == h and meta['plan_sha256'] == original.sha(original.PLAN)
            assert meta['request_sha256'] == original.digest(row) and meta['labels_read'] is False and meta['new_teacher_calls'] == 0
            with np.load(path, allow_pickle=False) as a:
                for name in a.files:
                    assert np.isfinite(a[name]).all(); finite += 1
                if bool(a['native_valid']):
                    with np.load(oldroot / oldseal['files'][q]['path'], allow_pickle=False) as old:
                        for name in ['Source', 'Native_interval', 'frame_ids', 'positions8']:
                            assert np.array_equal(a[name], old[name]), (ds, name)
                    for arm in ['Source', 'Spatial', 'Temporal', 'Full']:
                        assert (a[arm][:, 2:] > 0).all()
                    posterior = update(a['native_prior'], inclusive_scores(a['frame_feedback']))
                    assert hashlib.sha256(posterior.tobytes(order='C')).hexdigest() == meta['posterior_sha256']
                    restored += 1
                else:
                    assert meta['invalid_native_generation_retained']; invalid += 1
            manifest.update((q + ':' + h + ':' + original.sha(receipt) + '\n').encode())
            count += 1
        summaries[ds] = dict(committed_queries=count, total=len(rows), finite_arrays=finite,
                             posterior_exact_SHA_verified=restored, invalid_native_generations=invalid,
                             all_payload_receipt_request_plan_SHA_verified=True, valid_Source_bitexact_old=True,
                             labels_read=False, commit_manifest_sha256=manifest.hexdigest())
    assert summaries['vidstg']['committed_queries'] == 3949 and summaries['hc2']['committed_queries'] == 0
    result = dict(status='pass', scope='committed partial predictions only, no complete full score',
                  datasets=summaries, original_plan_sha256=original.sha(original.PLAN),
                  original_gate_sha256=original.sha(original.PUBLIC / 'both128_gate.json'),
                  TubeDETR_final_publication_commit=tube['commit'], GT_read=False)
    original.write(PREFLIGHT, result)


def prepare():
    assert not PLAN.exists()
    preflight()
    p = original.checked()
    gate = original.PUBLIC / 'both128_gate.json'
    files = [Path(__file__), original.PLAN, PREFLIGHT, gate, *[original.ROOT / name for name in json.loads(gate.read_text())['files']]]
    original.write(PLAN, dict(status='frozen_before_resuming_full',
        authorized_by='2026-10-26 user requests continuation, TubeDETR first and monitoring enabled',
        execution_order=stages(p), files={str(path.relative_to(original.ROOT)): original.sha(path) for path in files},
        completed_fixed128_not_rerun=True, original_inference_and_scoring_entry='tools.simplepath_llava.run',
        original_plan_sha256=original.sha(original.PLAN), original_gate_sha256=original.sha(gate),
        partial_predictions_verified_before_resume=True, parameters_and_source_readout_unchanged=True,
        new_teacher_calls=0, shared_gpu_lock=True, free_space_floor_GiB=25,
        frozen_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))


def checked():
    p = original.checked(); original.gate_both128(p)
    a = json.loads(PLAN.read_text()); assert a['status'] == 'frozen_before_resuming_full'
    for path, h in a['files'].items():
        assert original.sha(original.ROOT / path) == h, path
    assert a['execution_order'] == [list(x) for x in stages(p)]
    assert not (original.OUT / 'failure.json').exists()
    return p


def run():
    p = checked(); no_active_backbone()
    for stage, ds in stages(p):
        checked()
        cmd = [sys.executable, '-B', '-u', '-m', 'tools.simplepath_llava.run', stage, '--dataset', ds, '--full']
        log = original.ROOT / 'logs' / f'simplepath_llava_{ds}_full_{stage}_resume_v1.log'
        env = dict(os.environ)
        if stage == 'score':
            env['CUDA_VISIBLE_DEVICES'] = ''
        with log.open('ab') as stream:
            proc = subprocess.Popen(cmd, cwd=original.ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
            state = dict(status=stage, dataset=ds, full=True, plan_sha256=original.sha(original.PLAN),
                         resume_plan_sha256=original.sha(PLAN), controller_pid=os.getpid(), worker_pid=proc.pid,
                         log=str(log.relative_to(original.ROOT)))
            reg = original.ROOT / 'runtime/tubedetr_active_jobs.json'; jobs = json.loads(reg.read_text())
            jobs['jobs']['simplepath_llava_controller'] = dict(pid=os.getpid(), command=shlex.join([sys.executable, '-B', '-u', '-m', 'tools.simplepath_llava.resume', 'run']))
            jobs['jobs']['simplepath_llava_worker'] = dict(pid=proc.pid, command=shlex.join(cmd), stage=stage, dataset=ds, full=True, log=str(log.relative_to(original.ROOT)))
            original.write(reg, jobs); original.write(original.OUT / 'status.json', state)
            if proc.wait():
                raise RuntimeError(f'{ds} full {stage} failed: {log}')
    files = [original.PLAN, original.PUBLIC / 'qualification.json', original.PUBLIC / 'both128_gate.json',
             *[original.PUBLIC / ds / 'final_seal.json' for ds in p['dataset_order']]]
    original.write(original.PUBLIC / 'final_seal.json', dict(status='complete',
                   files={str(path.relative_to(original.ROOT)): original.sha(path) for path in files}, publication_pending=True))
    original.write(original.PUBLIC / 'resume_completion.json', dict(status='complete',
                   resume_plan_sha256=original.sha(PLAN), final_seal_sha256=original.sha(original.PUBLIC / 'final_seal.json')))
    original.write(original.OUT / 'status.json', dict(status='complete_both_fixed128_and_full',
                   plan_sha256=original.sha(original.PLAN), resume_plan_sha256=original.sha(PLAN)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except Exception as exc:
        original.write(original.OUT / 'resume_failure.json', dict(status='stopped', stage=args.stage,
                       error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
