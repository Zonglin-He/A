"""Isolated HC64 output-distribution adaptation; explicit --run, no GT access.

All three arms share one Source load and one sparse evidence construction per
query. Four output candidates use one differentiable student prediction, not
four decoder calls. The complete 64-query inference seal is the only handoff to
an independent offline scorer; this runner never imports or invokes scoring.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.state_opd.run import OLD, atomic, prepare, read, savept, sha
from scripts.output_opd.policy import ARMS, fit
from scripts.trajectory_fast.evidence import TrajectoryEvidence

HERE = Path(__file__).resolve().parent
TR = ROOT / 'outputs/output_opd_hc64'


def _inside_trial(relative):
    """Only receipt-bound relative paths within this independent trial."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Trial receipt contains an invalid relative path')
    result = TR / path
    if not result.resolve().is_relative_to(TR.resolve()):
        raise ValueError('Trial artifact resolves outside the independent trial')
    return result


def _verify_sources(lock):
    for section in ('sources', 'dependency_sources', 'frozen_sources', 'original_trajectory_sources'):
        for relative, expected in lock[section].items():
            assert sha(ROOT / relative) == expected, 'Locked source changed: ' + relative
    assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
    assert sha(ROOT / 'configs/default.json') == lock['registered_config_sha256']


def _snapshot(lock):
    """Retain source paths, avoiding collisions between different run.py files."""
    sources = {**lock['sources'], **lock['dependency_sources'], **lock['original_trajectory_sources']}
    for relative, expected in sources.items():
        source = ROOT / relative
        assert sha(source) == expected
        dest = TR / 'source_snapshot' / relative
        if dest.exists():
            assert sha(dest) == expected, 'Existing source snapshot differs'
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(dest.suffix + '.tmp')
            temp.write_bytes(source.read_bytes())
            os.replace(temp, dest)
    config_copy = TR / 'config.json'
    if config_copy.exists():
        assert config_copy.read_bytes() == (HERE / 'config.json').read_bytes()
    else:
        config_copy.write_bytes((HERE / 'config.json').read_bytes())


def _verify_receipt(key, receipt, item, fingerprint):
    assert receipt['GT_read'] is False and receipt['fingerprint'] == fingerprint
    assert receipt['reference_sha256'] == item['reference_sha256']
    assert receipt['cache_sha256'] == item['cache_sha256']
    assert receipt['native_mask_sha256'] == item['native_mask_sha256']
    assert set(receipt['costs']) == set(ARMS)
    dest = TR / 'predictions' / (key + '.npz')
    assert sha(dest) == receipt['sha256']
    assert sha(_inside_trial(receipt['trace_path'])) == receipt['trace_sha256']
    assert sha(_inside_trial(receipt['evidence_path'])) == receipt['evidence_sha256']
    with np.load(dest, allow_pickle=False) as data:
        assert set(data.files) == {'frame_ids', 'positions8', 'Strong_interval', *ARMS}
        assert data['positions8'].shape == (8,)
        for arm in ARMS:
            boxes = data[arm]
            assert boxes.shape == (len(data['frame_ids']), 4)
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
    for arm in ARMS:
        cost = receipt['costs'][arm]
        assert 1 <= cost['updates'] <= 3
        assert cost['full_trajectory_evals'] == 2 * cost['updates'] <= 6
        assert cost['native_offset_decoder_calls'] == 2 * cost['full_trajectory_evals']
        assert cost['decoder_backwards'] == cost['updates'] <= 3
        assert cost['setup_full_trajectory_evals'] == cost['setup_native_offset_decoder_calls'] == 0
        assert cost['output_candidates'] == (0 if arm == 'Direct' else 4 * cost['updates'])


def main(run=False, resume=False):
    _, records, items, oldlock = prepare()
    records, items = records[:64], items[:64]
    assert len(records) == len(items) == len({r['id'] for r in records}) == 64
    assert [r['id'] for r in records] == [item['id'] for item in items]
    cfg = read(HERE / 'config.json')
    assert cfg['queries'] == 64 and cfg['dataset'] == 'hc2'
    assert cfg['arms'] == list(ARMS) == ['OutputOPD', 'Direct', 'DirectMC']
    assert cfg['rounds'] == cfg['backward_budget'] == 3
    assert cfg['candidates'] == 4 and cfg['forward_budget'] == 6
    paths = sorted(p for p in HERE.glob('*.py') if not p.name.startswith('test_'))
    paths.append(HERE / 'config.json')
    dependency_paths = [ROOT / 'scripts/trajectory_fast/evidence.py',
                        ROOT / 'scripts/trajectory_fast/policy.py']
    original_lock_path = ROOT / 'outputs/trajectory_opd_hc64/lock.json'
    original_lock = read(original_lock_path)
    assert original_lock['GT_read'] is False
    identity = dict(
        config=cfg, config_sha256=sha(HERE / 'config.json'),
        parent_fingerprint=oldlock['fingerprint'],
        sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
        dependency_sources={str(p.relative_to(ROOT)): sha(p) for p in dependency_paths},
        frozen_sources=oldlock['sources'], loader_identity=oldlock['loader_identity'],
        formal_state_sha256=oldlock['formal_state_sha256'],
        registered_config_sha256=oldlock['registered_config_sha256'],
        manifest_sha256=oldlock['manifest_sha256'],
        original_trajectory_sources=original_lock['sources'],
        original_trajectory_lock_sha256=sha(original_lock_path),
        ordered_input_digest=hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
        ids=[r['id'] for r in records],
        selection='first64_original_manifest_order_one_query_per_parent',
        inference_only=True, GT_read=False)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    lock = {**identity, 'fingerprint': fingerprint}
    _verify_sources(lock)
    if (TR / 'lock.json').exists():
        assert read(TR / 'lock.json') == lock, 'Do not mix execution identities'
    else:
        atomic(TR / 'lock.json', lock)
    _snapshot(lock)
    if not run:
        print(json.dumps(dict(status='prepared_not_run', fingerprint=fingerprint,
                              queries=64, arms=ARMS, forward_budget=6,
                              backward_budget=3, candidates_per_sampled_round=4,
                              constructor_decoder_forwards=0, GT_read=False)))
        return
    assert not (TR / 'sealed.json').exists(), 'Already sealed, refuse overwrite'
    assert resume or not (TR / 'run.json').exists(), 'Existing execution requires explicit identical --resume'
    from stvg.runtime import lease, load_source, setup
    setup()
    leasefile = lease()
    model, files = None, {}
    begin = time.perf_counter()
    atomic(TR / 'run.json', dict(status='running', PID=os.getpid(),
                                 fingerprint=fingerprint, queries=64,
                                 started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                 GT_read=False))
    source_cfg = read(OLD / 'config.json')['datasets']['hc2']
    try:
        for record, item in zip(records, items):
            key = record['id']
            dest = TR / 'predictions' / (key + '.npz')
            receipt_path = dest.with_suffix('.json')
            if receipt_path.exists():
                receipt = read(receipt_path)
                _verify_receipt(key, receipt, item, fingerprint)
                files[key] = receipt
                continue
            assert not dest.exists(), 'Unreceipted prediction preserved; refuse overwrite'
            assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
            start = time.perf_counter()
            cache_path = OLD / 'student_cache' / (key + '.pt')
            mask_path = OLD / 'research/native_masks' / item['mask_path']
            reference_path = OLD / item['reference_path']
            assert sha(cache_path) == item['cache_sha256']
            assert sha(mask_path) == item['native_mask_sha256']
            assert sha(reference_path) == item['reference_sha256']
            cache = torch.load(cache_path, map_location='cpu', weights_only=False)
            pack = torch.load(mask_path, map_location='cpu', weights_only=False)
            assert cache['GT_read'] is pack['GT_read'] is False
            assert pack['source_cache_sha256'] == item['cache_sha256']
            with np.load(reference_path, allow_pickle=False) as data:
                gold = {name: data[name].copy() for name in data.files}
            assert np.array_equal(gold['E_Ref'], np.asarray(cache['E']))
            assert np.array_equal(gold['Source'], np.asarray(cache['Source']))
            assert gold['positions8'].tolist() == list(cache['planned_positions'])
            assert gold['frame_ids'].tolist() == list(cache['frame_ids'])
            assert np.array_equal(gold['Strong_interval'], np.asarray(cache['interval']))
            assert torch.equal(pack['teacher']['boxes'], cache['teacher']['boxes'])
            assert pack['selected_query'] == cache['teacher']['selected_query']
            if model is None:
                load_begin = time.perf_counter()
                model = load_source(source_cfg)
                model.cpu().eval().requires_grad_(False)
                gc.collect()
                torch.cuda.empty_cache()
                model.ground_decoder.decoder.cuda().eval().requires_grad_(False)
                atomic(TR / 'load.json', dict(seconds=time.perf_counter() - load_begin,
                                              source_loads_this_execution=1,
                                              encoder_forwards=0, expert_forwards=0,
                                              RGB_reads=0, GT_read=False))
            evidence_begin = time.perf_counter()
            evidence = TrajectoryEvidence(pack, cache, device='cuda')
            torch.cuda.synchronize()
            evidence_seconds = time.perf_counter() - evidence_begin
            result = {name: gold[name] for name in ('frame_ids', 'positions8', 'Strong_interval')}
            traces, costs = {}, {}
            for arm in ARMS:
                atomic(TR / 'progress.json', dict(completed=len(files), total=64,
                                                  id=key, arm=arm, phase='adapting', GT_read=False))
                out = fit(model, cache, evidence, item['key'], arm, cfg)
                assert out['arm'] == arm
                assert out['final'].shape == (len(cache['frame_ids']), 4)
                assert bool(torch.isfinite(out['final']).all())
                assert bool((out['final'][:, 2:] > 0).all())
                checks = out['initial_checks']
                assert checks['full_reference'] is True
                assert checks['constructor_decoder_calls'] == 0
                assert checks['original_native_Source_bit_exact'] is True
                assert checks['original_E_bit_exact'] is True
                assert checks['zero_identity_checked_on_first_charged_forward'] is True
                result[arm] = out['final'].detach().cpu().numpy()
                traces[arm], costs[arm] = out, out['costs']
                print(json.dumps(dict(completed=len(files), arm=arm, costs=out['costs'])), flush=True)
            evidence_path = TR / 'evidence' / (key + '.pt')
            trace_path = TR / 'traces' / (key + '.pt')
            assert not evidence_path.exists() and not trace_path.exists(), 'Preserve unreceipted artifacts'
            savept(evidence_path, evidence.private_state())
            savept(trace_path, traces)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix('.npz.tmp')
            with temp.open('wb') as handle:
                np.savez(handle, **result)
            os.replace(temp, dest)
            receipt = dict(
                fingerprint=fingerprint, sha256=sha(dest),
                trace_path=str(trace_path.relative_to(TR)), trace_sha256=sha(trace_path),
                evidence_path=str(evidence_path.relative_to(TR)), evidence_sha256=sha(evidence_path),
                evidence_summary=evidence.metadata, evidence_seconds=evidence_seconds,
                evidence_constructions=1, constructor_decoder_forwards=0,
                reference_sha256=item['reference_sha256'], cache_sha256=item['cache_sha256'],
                native_mask_sha256=item['native_mask_sha256'], costs=costs,
                total_seconds=time.perf_counter() - start, GT_read=False)
            _verify_receipt(key, receipt, item, fingerprint)
            atomic(receipt_path, receipt)
            files[key] = receipt
            atomic(TR / 'progress.json', dict(completed=len(files), total=64,
                                              seconds=receipt['total_seconds'], GT_read=False))
            print(json.dumps(dict(completed=len(files), seconds=receipt['total_seconds'])), flush=True)
            del traces, costs, out, result, gold, cache, pack, evidence
            gc.collect()
            torch.cuda.empty_cache()
        assert list(files) == [r['id'] for r in records] and len(files) == 64
        assert set((TR / 'predictions').glob('*.json')) == {
            TR / 'predictions' / (record['id'] + '.json') for record in records}
        assert set((TR / 'predictions').glob('*.npz')) == {
            TR / 'predictions' / (record['id'] + '.npz') for record in records}
        _verify_sources(lock)
        assert sha(original_lock_path) == lock['original_trajectory_lock_sha256']
        for record, item in zip(records, items):
            key = record['id']
            receipt = read(TR / 'predictions' / (key + '.json'))
            _verify_receipt(key, receipt, item, fingerprint)
        atomic(TR / 'sealed.json', dict(
            status='sealed_before_offline_scoring', fingerprint=fingerprint,
            dataset='hc2', queries=64, arms=list(ARMS),
            files={key: dict(path='predictions/' + key + '.npz', sha256=value['sha256'],
                             receipt_sha256=sha(TR / 'predictions' / (key + '.json')))
                   for key, value in files.items()},
            manifest_sha256=lock['manifest_sha256'], config_sha256=lock['config_sha256'],
            lock_sha256=sha(TR / 'lock.json'), GT_read=False))
        atomic(TR / 'run.json', dict(status='completed_sealed', fingerprint=fingerprint,
                                     queries=64, seconds=time.perf_counter() - begin, GT_read=False))
    except BaseException:
        atomic(TR / 'failure.json', dict(status='failed_preserved_no_skip',
                                         fingerprint=fingerprint, completed=len(files),
                                         traceback=traceback.format_exc(), GT_read=False))
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        leasefile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.run, args.resume)
