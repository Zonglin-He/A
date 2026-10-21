"""CPU-only propagation/matching HC64 run; no label or scorer imports.

All predictions, candidate banks, evidence and real update traces are sealed
before the independent scoring entry may access development labels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.state_opd.run import OLD, prepare, read, sha, atomic, savept
from scripts.trajectory_fast.evidence import TrajectoryEvidence

HERE = Path(__file__).resolve().parent
TRIAL = ROOT / 'outputs/propagation_opd_hc64'
ARMS = ('OPD', 'FixedKD', 'Direct')
PREDICTION_NAMES = ('FixedPropagation', *ARMS)


def validate_trace(traces, bank, predictions):
    """Verify recorded work and hard E fallback without model or GT access."""
    assert bank['GT_read'] is False
    active = torch.as_tensor(bank['active'], dtype=torch.bool)
    valid = torch.as_tensor(bank['valid'], dtype=torch.bool)
    E = torch.as_tensor(bank['E'])
    assert valid.shape == (len(E), 4) and bool(valid[:, 0].all())
    assert not bool(valid[~active, 1:].any())
    assert torch.equal(torch.as_tensor(bank['boxes'])[:, 0], E)
    assert set(traces) == set(PREDICTION_NAMES)
    for arm in PREDICTION_NAMES:
        output = traces[arm]; costs = output['costs']
        expected = 0 if arm == 'FixedPropagation' else 3
        assert costs['updates'] == costs['backward'] == costs['state_only_backwards'] == expected
        assert all(costs[name] == 0 for name in ('heavy_forwards', 'decoder_backwards',
                                               'decoder_calls', 'encoder_calls', 'expert_calls'))
        final = torch.as_tensor(output['final'])
        assert np.array_equal(predictions[arm], final.numpy())
        assert torch.equal(final[~active], E[~active]), (arm, 'inactive output changed')
        if arm == 'FixedPropagation':
            paths = [output['trace']]
            assert costs['total_sequence_rollouts'] == 1
        else:
            assert len(output['rounds']) == len(output['trace']['training_paths']) == 3
            assert all(len(paths) == 2 for paths in output['trace']['training_paths'])
            assert all(torch.as_tensor(row['gradient']).shape == (6,) and
                       bool(torch.isfinite(torch.as_tensor(row['gradient'])).all()) for row in output['rounds'])
            fixed = output['trace']['fixed_teacher_paths']
            assert len(fixed) == (2 if arm == 'FixedKD' else 0)
            assert costs['teacher_setup_rollouts'] == len(fixed)
            assert costs['student_training_rollouts'] == (0 if arm == 'FixedKD' else 6)
            assert costs['total_sequence_rollouts'] == (3 if arm == 'FixedKD' else 7)
            paths = [p for group in output['trace']['training_paths'] for p in group]
            paths += fixed + [output['trace']['final_rollout']]
        for path in paths:
            assert torch.equal(path['boxes'][~active], E[~active])
            assert not bool(path['actions'][~active].any())
            assert torch.equal(path['memory_before'][~active], path['memory_after'][~active])


def check_idle():
    """Check owned evaluation jobs only; never acquire GPU or signal a process."""
    formal = read(ROOT / 'outputs/formal/state.json')
    assert formal['status'] == 'paused_by_user', 'The original formal task must remain paused'
    blocked = []
    own = os.getpid()
    known = {'tools/run_formal.py', 'scripts/trajectory_opd/run.py', 'scripts/trajectory_fast/run.py',
             'scripts/output_opd/run.py', 'scripts/propagation_opd/run.py',
             'scripts/state_opd/run.py', 'scripts/state_tuning/run.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == own:
            continue
        try:
            args = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            cwd = Path(os.readlink(proc / 'cwd'))
        except (OSError, ProcessLookupError):
            continue
        for arg in args[1:]:
            if not arg.endswith('.py'):
                continue
            candidate = Path(arg)
            candidate = candidate.resolve() if candidate.is_absolute() else (cwd / candidate).resolve()
            if candidate.is_relative_to(ROOT) and str(candidate.relative_to(ROOT)) in known:
                blocked.append(dict(PID=int(proc.name), script=str(candidate.relative_to(ROOT))))
        if cwd.resolve() == ROOT and any(args[i:i+2] == ['-m', 'stvg'] for i in range(len(args) - 1)):
            blocked.append(dict(PID=int(proc.name), script='python -m stvg'))
    assert not blocked, f'Another owned evaluation worker is active: {blocked}'
    assert not torch.cuda.is_initialized(), 'This run must never initialize CUDA'


def source_files():
    required = ('bank.py', 'policy.py', 'run.py', 'score.py', 'config.json')
    assert all((HERE / name).is_file() for name in required), 'Finish all owned implementation files before locking'
    paths = [p for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    paths.append(HERE / 'config.json')
    paths.extend(ROOT / p for p in ('scripts/trajectory_fast/evidence.py',
                                  'scripts/trajectory_opd/evidence.py',
                                  'scripts/state_opd/run.py', 'scripts/state_opd/evaluate.py'))
    return sorted(set(paths))


def locked_inputs():
    _, records, items, parent = prepare()
    records, items = records[:64], items[:64]
    cfg = read(HERE / 'config.json')
    assert cfg['queries'] == 64 and cfg['dataset'] == 'hc2' and cfg['seed'] == 41
    assert cfg['rounds'] == 3 and cfg['paths'] == 2 and cfg['arms'] == list(ARMS)
    paths = source_files()
    identity = dict(config=cfg, sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
                    parent_fingerprint=parent['fingerprint'],
                    parent_lock_sha256=sha(ROOT / 'outputs/state_opd_hc128/lock.json'),
                    parent_all_inputs_digest=hashlib.sha256(json.dumps(parent['inputs'], sort_keys=True).encode()).hexdigest(),
                    frozen_sources=parent['sources'], loader_identity=parent['loader_identity'],
                    formal_state_sha256=parent['formal_state_sha256'],
                    registered_config_sha256=parent['registered_config_sha256'],
                    manifest_sha256=parent['manifest_sha256'],
                    reference_seal_sha256=parent['reference_seal_sha256'],
                    native_mask_seal_sha256=parent['native_mask_seal_sha256'],
                    inputs=items, ids=[record['id'] for record in records],
                    CPU_only=True, GT_read=False)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    lock = dict(identity, fingerprint=fingerprint)
    destination = TRIAL / 'lock.json'
    if destination.exists():
        assert read(destination) == lock, 'Do not mix execution/source/configuration fingerprints'
    else:
        atomic(destination, lock)
        for path in paths:
            copied = TRIAL / 'source_snapshot' / path.relative_to(ROOT)
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(path.read_bytes())
    return cfg, records, items, lock


def validate_receipt(key, item, lock):
    dest = TRIAL / 'predictions' / f'{key}.npz'
    receipt = read(dest.with_suffix('.json'))
    assert receipt['fingerprint'] == lock['fingerprint'] and receipt['GT_read'] is False
    assert sha(dest) == receipt['sha256']
    for stem in ('trace', 'bank', 'evidence'):
        path = (TRIAL / receipt[stem + '_path']).resolve()
        assert path.is_relative_to(TRIAL.resolve()) and sha(path) == receipt[stem + '_sha256']
    for field in ('cache_sha256', 'native_mask_sha256', 'reference_sha256'):
        assert receipt[field] == item[field]
    bank = torch.load(TRIAL / receipt['bank_path'], map_location='cpu', weights_only=False)
    traces = torch.load(TRIAL / receipt['trace_path'], map_location='cpu', weights_only=False)
    with np.load(dest, allow_pickle=False) as saved:
        predictions = {name: saved[name].copy() for name in PREDICTION_NAMES}
    validate_trace(traces, bank, predictions)
    return receipt


def main(run=False, resume=False):
    cfg, records, items, lock = locked_inputs()
    if not run:
        print(json.dumps(dict(status='prepared_not_run', fingerprint=lock['fingerprint'], queries=64,
                              arms=list(PREDICTION_NAMES), CPU_only=True, GT_read=False)))
        return
    check_idle()
    assert not (TRIAL / 'sealed.json').exists(), 'Already sealed; refuse overwriting'
    assert resume or not (TRIAL / 'run.json').exists(), 'Existing execution requires identical explicit --resume'
    from scripts.propagation_opd.bank import CandidateBank
    from scripts.propagation_opd.policy import ARMS as POLICY_ARMS, fit, fixed_propagation
    assert tuple(POLICY_ARMS) == ARMS
    torch.set_num_threads(1)
    begin = time.perf_counter()
    files = {}
    atomic(TRIAL / 'run.json', dict(status='running', PID=os.getpid(), fingerprint=lock['fingerprint'],
                                   CPU_only=True, GT_read=False))
    try:
        for record, item in zip(records, items):
            key = record['id']
            dest = TRIAL / 'predictions' / f'{key}.npz'
            if dest.with_suffix('.json').exists():
                files[key] = validate_receipt(key, item, lock)
                continue
            assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
            assert sha(ROOT / 'configs/default.json') == lock['registered_config_sha256']
            start = time.perf_counter()
            cache_path = OLD / 'student_cache' / f'{key}.pt'
            mask_path = OLD / 'research/native_masks' / item['mask_path']
            reference_path = OLD / item['reference_path']
            assert sha(cache_path) == item['cache_sha256']
            assert sha(mask_path) == item['native_mask_sha256']
            assert sha(reference_path) == item['reference_sha256']
            cache = torch.load(cache_path, map_location='cpu', weights_only=False)
            pack = torch.load(mask_path, map_location='cpu', weights_only=False)
            assert cache['GT_read'] is pack['GT_read'] is False
            with np.load(reference_path, allow_pickle=False) as saved:
                gold = {name: saved[name].copy() for name in saved.files}
            assert np.array_equal(gold['E_Ref'], cache['E']) and np.array_equal(gold['Source'], cache['Source'])
            evidence_begin = time.perf_counter()
            evidence = TrajectoryEvidence(pack, cache, device='cpu')
            evidence_seconds = time.perf_counter() - evidence_begin
            bank_begin = time.perf_counter()
            bank = CandidateBank(evidence, cache, cfg)
            bank_seconds = time.perf_counter() - bank_begin
            result = {name: gold[name] for name in ('frame_ids', 'positions8', 'Strong_interval')}
            result.update(SourceStrong=gold['Source'], ERefStrong=gold['E_Ref'], GeometryOPD=gold['GeometryOPD'])
            traces, costs = {}, {}
            for arm in PREDICTION_NAMES:
                adapted = fixed_propagation(bank, cfg) if arm == 'FixedPropagation' else fit(bank, item['key'], arm, cfg)
                final = adapted['final']
                assert torch.is_tensor(final) and final.device.type == 'cpu'
                assert final.shape == (len(gold['frame_ids']), 4)
                assert bool(torch.isfinite(final).all()) and bool((final[:, 2:] > 0).all())
                result[arm] = final.detach().numpy().copy()
                traces[arm], costs[arm] = adapted, adapted['costs']
                assert not torch.cuda.is_initialized()
                atomic(TRIAL / 'progress.json', dict(completed=len(files), total=64, id=key, arm=arm,
                                                    CPU_only=True, GT_read=False))
            evidence_path = TRIAL / 'evidence' / f'{key}.pt'
            bank_path = TRIAL / 'banks' / f'{key}.pt'
            trace_path = TRIAL / 'traces' / f'{key}.pt'
            savept(evidence_path, evidence.private_state())
            savept(bank_path, bank.private_state())
            savept(trace_path, traces)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix('.npz.tmp')
            with temporary.open('wb') as handle:
                np.savez(handle, **result)
            os.replace(temporary, dest)
            receipt = dict(fingerprint=lock['fingerprint'], sha256=sha(dest),
                           trace_path=str(trace_path.relative_to(TRIAL)), trace_sha256=sha(trace_path),
                           bank_path=str(bank_path.relative_to(TRIAL)), bank_sha256=sha(bank_path),
                           evidence_path=str(evidence_path.relative_to(TRIAL)), evidence_sha256=sha(evidence_path),
                           evidence_summary=evidence.metadata, bank_summary=bank.metadata,
                           evidence_seconds=evidence_seconds, bank_seconds=bank_seconds,
                           reference_sha256=item['reference_sha256'], cache_sha256=item['cache_sha256'],
                           native_mask_sha256=item['native_mask_sha256'], costs=costs,
                           total_seconds=time.perf_counter() - start,
                           model_calls=0, decoder_calls=0, expert_calls=0, RGB_reads=0,
                           CPU_only=True, GT_read=False)
            atomic(dest.with_suffix('.json'), receipt)
            files[key] = validate_receipt(key, item, lock)
            atomic(TRIAL / 'progress.json', dict(completed=len(files), total=64, CPU_only=True, GT_read=False))
            print(json.dumps(dict(completed=len(files), total=64, seconds=receipt['total_seconds'], costs=costs)), flush=True)
            del cache, pack, evidence, bank, traces, result, adapted, final
            gc.collect()
        assert len(files) == 64
        # Revalidate the entire cohort after all disk writes, not only each
        # artifact at the time it was produced or accepted for resume.
        for record, item in zip(records, items):
            files[record['id']] = validate_receipt(record['id'], item, lock)
        for path, digest in {**lock['frozen_sources'], **lock['sources']}.items():
            assert sha(ROOT / path) == digest, path
        assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
        assert sha(ROOT / 'configs/default.json') == lock['registered_config_sha256']
        assert not torch.cuda.is_initialized()
        atomic(TRIAL / 'sealed.json', dict(fingerprint=lock['fingerprint'], queries=64, dataset='hc2',
            lock_sha256=sha(TRIAL / 'lock.json'), manifest_sha256=lock['manifest_sha256'],
            files={key: dict(path='predictions/' + key + '.npz', sha256=value['sha256'],
                            receipt_sha256=sha(TRIAL / 'predictions' / f'{key}.json')) for key, value in files.items()},
            CPU_only=True, GT_read=False))
        atomic(TRIAL / 'run.json', dict(status='completed_sealed', fingerprint=lock['fingerprint'],
            queries=64, seconds=time.perf_counter() - begin, CPU_only=True, GT_read=False))
    except BaseException:
        atomic(TRIAL / 'failure.json', dict(completed=len(files), fingerprint=lock['fingerprint'],
                                            traceback=traceback.format_exc(), CPU_only=True, GT_read=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.run, args.resume)
