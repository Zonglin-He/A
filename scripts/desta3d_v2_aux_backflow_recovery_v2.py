"""Registered isolated B replay from valid COMMON_A after optimizer-ID failure.

The original locked worker stays byte-identical. Only checkpoint I/O and the
initialization entry point are replaced here, before starting this new process.
The original A/B observations and the failed resume remain untouched.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import desta3d_v2_aux_backflow_fit as worker
from vg_tta.optimizer_checkpoint import cpu_clone, restore_optimizer, validate_serialized_optimizer

BASE = ROOT/'artifacts/desta3d_v2/aux_backflow_v1'
PRIOR = BASE/'fit'
OUT = BASE/'fit_recovery_v2'
COMMON_SHA = 'ef58d4b4e53f2e2cd77c8a8647a7c017291168a1b35c622adac29999c7dc2822'


def atomic_pt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pt')
    torch.save(cpu_clone(value), temporary)
    os.replace(temporary, path)


def save_state(state, adapters, optimizers):
    payload = state | {
        'adapters': {a: cpu_clone(m.state_dict()) for a, m in adapters.items()},
        'optimizers': {a: cpu_clone(o.state_dict()) for a, o in optimizers.items()},
        'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'lock_sha': worker.sha(OUT/'LOCK.json'), 'checkpoint_schema': 'integer_optimizer_ids_v2',
    }
    for opt in payload['optimizers'].values():
        validate_serialized_optimizer(opt)
    atomic_pt(OUT/'LATEST.pt', payload)


def initialize():
    assert worker.sha(PRIOR/'COMMON_A.pt') == COMMON_SHA
    common = torch.load(PRIOR/'COMMON_A.pt', map_location='cpu', weights_only=False)['common']
    torch.manual_seed(worker.SEED)
    torch.cuda.manual_seed_all(worker.SEED)
    adapters, optimizers = {}, {}
    for arm in worker.MODES:
        model = worker.Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda()
        model.load_state_dict(common)
        adapters[arm] = model
        optimizers[arm] = worker.make_source_optimizer(model, 'repaired', 'B')
    hashes = {a: worker.adapter_sha256(m) for a, m in adapters.items()}
    assert hashes == worker.read(PRIOR/'COMMON_A.json')['arm_hashes']
    state = {'stage': 'B', 'cursor': 0, 'steps': dict.fromkeys(worker.MODES, 0), 'last_window': None}
    save_state(state, adapters, optimizers)
    worker.write(OUT/'INITIAL.json', {'adapter_hashes': hashes, 'common_A_sha': COMMON_SHA,
        'fresh_B_optimizers': True, 'seed': worker.SEED, 'A_retrained': False,
        'replay_B_from_common_A_not_resume_B344': True,
        'original_B_initial_RNG_not_retained': True, 'bitwise_original_B_replay_claim': False})
    return state, adapters, optimizers


def restore():
    saved = torch.load(OUT/'LATEST.pt', map_location='cpu', weights_only=False)
    assert saved['lock_sha'] == worker.sha(OUT/'LOCK.json')
    assert saved['checkpoint_schema'] == 'integer_optimizer_ids_v2'
    assert saved['stage'] in ('B', 'validation')
    adapters, optimizers = {}, {}
    for arm, weights in saved['adapters'].items():
        model = worker.Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda()
        model.load_state_dict(weights)
        opt = worker.make_source_optimizer(model, 'repaired', 'B')
        restore_optimizer(opt, saved['optimizers'][arm])
        adapters[arm], optimizers[arm] = model, opt
    state = {k: v for k, v in saved.items() if k not in (
        'adapters', 'optimizers', 'cpu_rng', 'cuda_rng', 'lock_sha', 'checkpoint_schema')}
    torch.set_rng_state(saved['cpu_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    worker.commit_history(state)
    worker.write(OUT/'restore_entries'/f'{time.time_ns()}.json', {
        'checkpoint_sha': worker.sha(OUT/'LATEST.pt'), 'stage': state['stage'],
        'cursor': state['cursor'], 'steps': state['steps'],
        'state_bound_to_live_parameters': True, 'state_exact_equal_after_load': True,
        'optimizer_steps': {a: sorted({int(v['step']) for v in o.state.values()})
                            for a, o in optimizers.items()},
        'source_validation_GT_read': False, 'target_GT_read': False})
    return state, adapters, optimizers


def configure_worker():
    worker.OUT = OUT
    worker.cpu_copy = cpu_clone
    worker.atomic_pt = atomic_pt
    worker.save_state = save_state
    worker.initialize = initialize
    worker.restore = restore


def register():
    assert not (OUT/'LOCK.json').exists()
    assert worker.sha(PRIOR/'COMMON_A.pt') == COMMON_SHA
    checks = worker.read(BASE/'OPTIMIZER_RECOVERY_CPU_CHECK.json')
    assert checks['status'] == 'passed'
    config = worker.read(PRIOR/'CONFIG.json') | {
        'recovery': 'restart only B from immutable valid COMMON_A, fixed paired three-arm comparison',
        'A_epochs_executed_in_this_directory': 0, 'B_epochs': 1,
        'common_A_sha': COMMON_SHA, 'checkpoint_schema': 'integer_optimizer_ids_v2',
        'fresh_B_seed': worker.SEED, 'bitwise_original_B_replay_claim': False,
        'previous_86_B_windows_need_replay': True,
        'initialization_scope': 'same A weights and recipe; new explicit seeded B RNG, all arms paired',
    }
    worker.write(OUT/'CONFIG.json', config)
    old_lock = worker.read(PRIOR/'LOCK.json')
    for path, digest in old_lock['pins'].items():
        assert worker.sha(path) == digest, path
    pins = dict(old_lock['pins'])
    for path in [Path(__file__), ROOT/'vg_tta/optimizer_checkpoint.py',
                 ROOT/'tests/test_desta3d_optimizer_checkpoint.py',
                 BASE/'OPTIMIZER_RECOVERY_CPU_CHECK.json',
                 BASE/'OPTIMIZER_RESTORE_INCIDENT_20260928.json', PRIOR/'COMMON_A.pt',
                 OUT/'CONFIG.json', ROOT/'protocols/desta3d_v2_optimizer_recovery_v2.md']:
        pins[str(path)] = worker.sha(path)
    worker.write(OUT/'LOCK.json', {'time': time.time(), 'pins': pins})
    # Immutable hardlink avoids duplicating the valid evidence checkpoint.
    os.link(PRIOR/'COMMON_A.pt', OUT/'COMMON_A.pt')
    worker.write(OUT/'COMMON_A.json', worker.read(PRIOR/'COMMON_A.json') | {
        'reused_valid_evidence_checkpoint': True, 'A_retrained': False})
    worker.write(OUT/'REGISTRATION.json', {'time': time.time(), 'config_sha': worker.sha(OUT/'CONFIG.json'),
        'lock_sha': worker.sha(OUT/'LOCK.json'), 'scope': 'same B0/B1/B2 source contrast with correct optimizer persistence',
        'prior_failed_resume_preserved': True, 'source_validation_GT_read': False, 'target_GT_read': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['register', 'fit'])
    ap.add_argument('--allocation', default='recovery001')
    ap.add_argument('--phase-seconds', type=float, default=3600)
    args = ap.parse_args()
    if args.action == 'register':
        return register()
    for path, digest in worker.read(OUT/'LOCK.json')['pins'].items():
        assert worker.sha(path) == digest, path
    assert not (OUT/'PREDICTIONS_COMPLETE.json').exists()
    lease = (ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = ROOT/'artifacts/desta3d_v2/receipts'/f'aux_backflow_{args.allocation}.json'
    assert not receipt.exists() and not (OUT/'starts'/f'{args.allocation}.json').exists()
    # Protect each allocation entry point before LATEST is ever replaced.
    resume_sha = None
    if (OUT/'LATEST.pt').exists():
        snapshot = OUT/'resume_points'/f'{args.allocation}.pt'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        os.link(OUT/'LATEST.pt', snapshot)
        resume_sha = worker.sha(snapshot)
    configure_worker()
    began = time.monotonic(); status = 'running'; failure = None
    def guard():
        if shutil.disk_usage(ROOT).free < 8 * 2**30:
            raise RuntimeError('8GiB disk reserve')
        return time.monotonic() - began > args.phase_seconds - 30
    try:
        assert not guard()
        torch.set_num_threads(4)
        torch.manual_seed(worker.SEED); torch.cuda.manual_seed_all(worker.SEED)
        torch.cuda.reset_peak_memory_stats()
        worker.write(OUT/'starts'/f'{args.allocation}.json', {
            'pid': os.getpid(), 'time': time.time(), 'phase_seconds': args.phase_seconds,
            'prior_seconds': sum(worker.scan_nested_gpu_receipts(p)[0] for p in (worker.V1, worker.OLD.parent)),
            'cumulative_cap_seconds': None, 'resume_checkpoint_sha': resume_sha})
        status = worker.fit(guard)
    except BaseException:
        failure = traceback.format_exc(); status = 'failed'; raise
    finally:
        worker.write(receipt, {'allocation': args.allocation, 'stage': 'v2_aux_backflow_recovery_source_contrast',
            'seconds': time.monotonic() - began, 'status': status, 'failure': failure,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'cumulative_cap_seconds': None})


if __name__ == '__main__':
    main()
