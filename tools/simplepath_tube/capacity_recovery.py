"""Explicit Root recovery: preserve bounded caches and resume identical inference.

The original frozen engine, numerical adapter, plan, locks and prior predictions
are immutable. Only redundant cache persistence becomes retention-only. Every
prediction still saves its original native logits/prior and all five outputs.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import numpy as np
from stvg.runtime import ROOT, sha
from stvg.tube_runner import write, digest
from tools.reuse_cache.store import ArrayStore
from . import run as original

PLAN = original.PUBLIC / 'capacity_recovery_plan.json'
CHECKS = original.PUBLIC / 'capacity_recovery_checks.json'
ARCHIVE = ROOT / 'outputs/simplepath_tubedetr_capacity_recovery/cache64_v1'
FAILURE = original.OUT / 'capacity_recovery_failure.json'


class RetentionOnlyStore(ArrayStore):
    """Read/validate existing cache identities; explicitly skip new duplicates."""
    amendment_sha256 = None

    def put(self, context, arrays, *, required=False):
        key = self.key(context)
        if (self.root / (key + '.json')).exists() or (self.root / (key + '.npz')).exists():
            return super().put(context, arrays, required=True)
        values = {k: np.asarray(v) for k, v in arrays.items()}
        if not values or any(not k.isidentifier() for k in values):
            raise ValueError('Named numerical arrays required')
        if any(a.dtype.hasobject or a.dtype.kind not in 'biufc' or not np.isfinite(a).all()
               for a in values.values()):
            raise ValueError('Finite numerical cache candidates required')
        return dict(status='not_cached_capacity_recovery_retention_only', key=key,
                    execution_amendment_sha256=self.amendment_sha256,
                    arrays_retained_in_current_inference=True,
                    native_outputs_and_final_prediction_saved_separately=True)


def prepare():
    if PLAN.exists() or ARCHIVE.exists():
        raise ValueError('Recovery amendment already frozen; never replace')
    p = original.checked()
    original.gate_both128(p)
    failure = original.OUT / 'failure.json'
    assert json.loads(failure.read_text())['status'] == 'stopped'
    log = ROOT / 'logs/simplepath_tubedetr_vidstg_full_gpu.log'
    assert 'CapacityError: Cache budget/reserve reached' in log.read_text()[-6000:]
    jobs = json.loads((ROOT / 'runtime/tubedetr_active_jobs.json').read_text())['jobs']
    for key in ['simplepath_tubedetr_controller', 'simplepath_tubedetr_worker']:
        proc = Path('/proc') / str(jobs[key]['pid']) / 'cmdline'
        if proc.exists() and b'tools.simplepath_tube' in proc.read_bytes():
            raise ValueError('Prior experiment process still active')
    target = original.OUT / 'vidstg/full'
    rows = original.rows_for(p, 'vidstg', True)
    prefix = []
    for row in rows:
        path = target / (row['id'] + '.npz'); receipt = path.with_suffix('.json')
        if not path.exists() and not receipt.exists():
            break
        assert path.exists() and receipt.exists()
        meta = json.loads(receipt.read_text())
        assert meta['plan_sha256'] == sha(original.PLAN) and meta['request_sha256'] == digest(row)
        assert meta['labels_read'] is False and meta['new_teacher_calls'] == 0
        prefix.append(dict(request_sha256=digest(row), payload_sha256=meta['payload_sha256'],
                           receipt_sha256=sha(receipt), bytes=path.stat().st_size))
    assert len(prefix) == len(list(target.glob('*.npz'))) == 8591
    assert not list(target.glob('*.pending'))
    ARCHIVE.mkdir(parents=True)
    files = [failure, original.OUT/'status.json', original.OUT/'worker_progress.json',
             log, ROOT/'logs/complete_simplepath_tubedetr.log',
             ROOT/'tools/simplepath_tube/run.py']
    archived = {}
    for f in files:
        dst = ARCHIVE / f.name
        shutil.copy2(f, dst); assert sha(f) == sha(dst)
        archived[str(dst.relative_to(ROOT))] = sha(dst)
    write(ARCHIVE/'committed_prefix.json', prefix)
    write(PLAN, dict(status='frozen_engineering_capacity_amendment_before_resumption',
        original_run_plan_sha256=sha(original.PLAN), fixed_method_lock_sha256=sha(original.PUBLIC/'fixed_method_lock.json'),
        both128_gate_sha256=sha(original.PUBLIC/'both128_gate.json'),
        source_sha256=sha(Path(__file__)), tests_sha256=sha(ROOT/'tests/test_simplepath_capacity_recovery.py'),
        verified_CPU_checks_sha256=sha(CHECKS), archived_files=archived,
        committed_prefix_queries=len(prefix), original_population_sizes=dict(vidstg=10303,hc2=1902),
        scientific_protocol_unchanged=True, inference_code_and_weights_unchanged=True,
        no_parameter_query_or_expert_change=True, new_teacher_calls=0,
        cache_policy='keep every existing payload; validate exact hits; no new cache appends; compute cold visual prefix when absent',
        prediction_policy='every final NPZ/receipt still required; all original native logits/prior and five arms retained',
        budget_bytes=64 << 30, minimum_free_bytes=25 << 30, no_deletion=True,
        old_failure_retained_as_historical=True, automatic_retry=False,
        created_at=datetime.datetime.now().astimezone().isoformat()))
    print('capacity amendment frozen', sha(PLAN), flush=True)


def checked():
    p = json.loads(PLAN.read_text())
    assert sha(original.PLAN) == p['original_run_plan_sha256']
    assert sha(Path(__file__)) == p['source_sha256']
    assert sha(ROOT/'tests/test_simplepath_capacity_recovery.py') == p['tests_sha256']
    assert sha(CHECKS) == p['verified_CPU_checks_sha256']
    for name,h in p['archived_files'].items():
        assert sha(ROOT/name) == h, name
    original.checked(); original.gate_both128(None)
    return p


def gpu(ds):
    checked()
    RetentionOnlyStore.amendment_sha256 = sha(PLAN)
    original.ArrayStore = RetentionOnlyStore
    original.gpu(ds, True)
    path = original.OUT/ds/'full/sealed.json'
    s = json.loads(path.read_text())
    s.update(execution_amendment_sha256=sha(PLAN), cache_append_disabled=True,
             committed_outputs_resumed_without_replacement=True)
    write(path,s)


def score(ds):
    checked(); original.score(ds, True)
    path = original.PUBLIC/ds/'final_seal.json'
    s = json.loads(path.read_text())
    s['files'].update({str(f.relative_to(ROOT)):sha(f) for f in [PLAN,CHECKS]})
    write(path,s)


def child(stage,ds):
    cmd=[sys.executable,'-B','-u','-m','tools.simplepath_tube.capacity_recovery',stage,'--dataset',ds]
    log=ROOT/'logs'/f'simplepath_tubedetr_{ds}_full_{stage}_capacity_v1.log'
    write(original.OUT/'status.json',dict(status=stage,dataset=ds,full=True,
          plan_sha256=sha(original.PLAN),execution_amendment_sha256=sha(PLAN),
          historical_failure_resolved_by_explicit_Root_recovery=True))
    with log.open('ab') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
              env={**os.environ,'CUDA_VISIBLE_DEVICES':''} if stage=='score' else os.environ.copy())
        reg=ROOT/'runtime/tubedetr_active_jobs.json';jobs=json.loads(reg.read_text())
        jobs['jobs']['simplepath_tubedetr_controller']=dict(pid=os.getpid(),
             command=shlex.join([sys.executable,'-B','-u','-m','tools.simplepath_tube.capacity_recovery','run']),
             log='logs/complete_simplepath_tubedetr_capacity_v1.log',execution_amendment_sha256=sha(PLAN))
        jobs['jobs']['simplepath_tubedetr_worker']=dict(pid=proc.pid,command=shlex.join(cmd),stage=stage,
             dataset=ds,full=True,log=str(log.relative_to(ROOT)),execution_amendment_sha256=sha(PLAN))
        write(reg,jobs)
        if proc.wait():raise RuntimeError('Capacity recovery stage failed: '+str(log))


def run():
    checked()
    if FAILURE.exists():raise ValueError('Recovery failure retained; no automatic retry')
    for ds in ['vidstg','hc2']:
        child('gpu',ds);child('score',ds)
    files=[original.PLAN,original.PUBLIC/'qualification.json',original.PUBLIC/'both128_gate.json',
           PLAN,CHECKS,*[original.PUBLIC/ds/'final_seal.json' for ds in ['vidstg','hc2']]]
    write(original.PUBLIC/'final_seal.json',dict(status='complete',
          files={str(f.relative_to(ROOT)):sha(f) for f in files},publication_pending=True))
    write(original.OUT/'status.json',dict(status='complete_both_fixed128_and_full',
          plan_sha256=sha(original.PLAN),execution_amendment_sha256=sha(PLAN),
          historical_failure_resolved_by_explicit_Root_recovery=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','gpu','score'])
    parser.add_argument('--dataset',choices=['vidstg','hc2']);args=parser.parse_args()
    try:
        if args.stage in ['prepare','run']:globals()[args.stage]()
        else:globals()[args.stage](args.dataset)
    except Exception as error:
        write(FAILURE,dict(status='stopped',stage=args.stage,dataset=args.dataset,
              error=type(error).__name__+': '+str(error),automatic_retry=False))
        raise
