"""CPU-only, write-once audit of a completed locked mixer seed.

This checks training coverage/state; it does not score native utility or inspect
the running worker's memory. Frozen PTD/B1 scope is supported by locked worker
assertions and saved hashes, not a new full-backbone comparison here.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D = ROOT / 'artifacts/desta3d_v3/latent_oracle_v1/joint_learnability_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(seed):
    import torch
    from vg_tta.desta3d_v3_joint_mixer import JointCorrectionMixer
    from vg_tta.optimizer_checkpoint import assert_nested_equal, restore_optimizer
    from vg_tta.optimizer_checkpoint import validate_serialized_optimizer

    torch.set_num_threads(4)
    start = time.monotonic()
    dest = D / f'seed{seed}'
    out = dest / 'ROOT_TRAIN_AUDIT_V2.json'
    assert not out.exists(), 'Do not overwrite a completed audit'
    cfg = read(D / 'CONFIG.json')
    assert seed in cfg['seeds']
    complete = read(dest / 'COMPLETE.json')
    load = lambda p: torch.load(p, map_location='cpu', weights_only=False)
    final, initial = load(dest / 'FINAL.pt'), load(dest / 'INITIAL.pt')
    assert sha(dest / 'FINAL.pt') == complete['final_sha']
    assert final['seed'] == seed and initial['seed'] == seed
    assert final['lock_sha'] == initial['lock_sha'] == sha(D / 'LOCK.json')
    assert (final['cursor'], final['windows']) == (618, 155)
    assert initial['steps'] == initial['cursor'] == 0
    basis = load(D / 'BASIS.pt')
    assert torch.equal(basis, final['mixer']['basis'])
    assert torch.equal(initial['mixer']['basis'], basis)
    assert torch.count_nonzero(initial['mixer']['output.weight']) == 0
    assert torch.count_nonzero(initial['mixer']['output.bias']) == 0
    mixer = JointCorrectionMixer(basis, radius=cfg['radius'])
    mixer.load_state_dict(final['mixer'], strict=True)
    assert sum(p.numel() for p in mixer.parameters()) == 103424
    assert not list(mixer.buffers())[0].requires_grad
    assert all(torch.isfinite(v).all() for v in final['mixer'].values())
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=cfg['lr'], weight_decay=0)
    validate_serialized_optimizer(final['optimizer'])
    restore_optimizer(optimizer, final['optimizer'])
    params = {p for group in optimizer.param_groups for p in group['params']}
    assert all(isinstance(p, torch.nn.Parameter) and p in params for p in optimizer.state)
    counters = [int(s['step']) for s in optimizer.state.values()]
    assert len(counters) == 8 and set(counters) == {final['steps']}
    assert all(torch.isfinite(s[k]).all() for s in optimizer.state.values()
               for k in ('exp_avg', 'exp_avg_sq'))
    assert final['cpu_rng'].dtype == torch.uint8 and len(final['cuda_rng']) == 1
    rows = read(D / 'TRAIN_INPUTS.json')
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    histories = sorted((dest / 'history').glob('*.json'))
    assert len(histories) == 155
    seen, parents, keys, missing = [], set(), [], {'event': {}, 'spatial': {}}
    frozen_hashes, union_hashes = set(), set()
    clips = empty = steps = zero_lr_windows = format_failures = early_event_failures = 0
    min_ratio, max_ratio = math.inf, -math.inf
    previous_cursor = 0
    last = None
    for window, path in enumerate(histories, 1):
        h = read(path)
        assert path.name == f'{window:04}.json' and h['windows'] == window and h['seed'] == seed
        stop = min(previous_cursor + cfg['accum'], len(rows))
        assert h['cursor'] == stop and len(h['queries']) == stop - previous_cursor
        assert h['steps'] - steps in (0, 1)
        assert h['empty_windows'] == window - h['steps']
        assert h['counters'] == [h['steps']] * 8
        expected_lr = cfg['lr'] * (window / 8 if window <= 8 else
                                   .5 * (1 + math.cos(math.pi * (window - 8) / (155 - 8))))
        assert h['lr'] == expected_lr and math.isfinite(h['gradient_norm'])
        assert h['clip'] == (h['gradient_norm'] > cfg['clip'])
        clips += h['clip']
        zero_lr_windows += h['lr'] == 0
        frozen_hashes.add(h['frozen_adapter_sha'])
        union_hashes.add(h['union_sha'])
        for position, q in zip(range(previous_cursor, stop), h['queries']):
            expected = rows[order[position]]
            assert q['position'] == position and q['key'] == expected['key']
            assert q['source'] == expected['source']
            seen.append(position); keys.append(q['key']); parents.add(q['source'])
            inj = q['injection']
            assert inj['same_field_both_passes']
            if inj['calls'] == ['event']:
                # Original decoder explicitly retains event failure and does
                # not start spatial. The protocol retains missing support.
                assert not q['native_format']
                assert q['branches']['event'].get('missing') == 'missing_native_event_support'
                assert q['branches']['spatial'].get('missing') == 'missing_native_spatial_support'
                early_event_failures += 1
            else:
                assert inj['calls'] == ['event', 'spatial']
            ratio = inj['relative_norm']
            assert math.isfinite(ratio) and 0 <= ratio <= cfg['radius'] + 2e-6
            min_ratio, max_ratio = min(min_ratio, ratio), max(max_ratio, ratio)
            format_failures += not q['native_format']
            for branch in ('event', 'spatial'):
                b = q['branches'][branch]
                if 'missing' in b:
                    reason = b['missing']
                    missing[branch][reason] = missing[branch].get(reason, 0) + 1
                else:
                    assert math.isfinite(b['CE']) and b['actions'] > 0
                    if branch == 'spatial':
                        assert b['classes'] == 152775
                    else:
                        assert b['classes'] == q['physical_pixel']['grid'][0][0]
                        assert b['actions'] == 2
        previous_cursor, steps, empty, last = stop, h['steps'], h['empty_windows'], h
    assert seen == list(range(618)) and len(set(keys)) == 618 and len(parents) == 95
    assert len(frozen_hashes) == len(union_hashes) == 1
    from scripts.desta3d_v3_privileged_ptd_qualification import ADAPTER_SHA
    from vg_tta.desta3d_v3_oracle_io import tensor_sha
    assert frozen_hashes == {ADAPTER_SHA} and union_hashes == {tensor_sha(basis)}
    assert_nested_equal(final['last_history'], last)
    assert final['steps'] == steps and final['empty_windows'] == empty
    changed = [k for k in final['mixer'] if not torch.equal(final['mixer'][k], initial['mixer'][k])]
    assert set(changed) <= {name for name, _ in mixer.named_parameters()}
    evidence = [dest / 'FINAL.pt', dest / 'INITIAL.pt', dest / 'COMPLETE.json',
                D / 'LOCK.json', D / 'CONFIG.json', D / 'TRAIN_INPUTS.json', D / 'BASIS.pt', *histories]
    result = dict(status='passed', seed=seed, queries=618, parents=95, windows=155,
                  actual_adam_steps=steps, empty_windows=empty, counters=counters,
                  optimizer_integer_keys=True, CPU_restored_live_parameter_binding=True,
                  basis_exact=True, changed_mixer_tensors=changed, parameters=103424,
                  clip_windows=clips, zero_learning_rate_windows=zero_lr_windows,
                  missing_action_support=missing, training_native_format_failures=format_failures,
                  retained_early_event_failures=early_event_failures,
                  relative_injection_norm_range=[min_ratio, max_ratio],
                  deterministic_query_order_and_coverage=True,
                  last_window_queries=len(last['queries']),
                  final_sha=complete['final_sha'],
                  evidence_sha256={str(p.relative_to(ROOT)): sha(p) for p in evidence},
                  auditor_sha=sha(Path(__file__)), CPU_seconds=time.monotonic()-start,
                  utility_scored=False, target_read=False,
                  limitations='Saved-state CPU restore and full query-history checks; frozen PTD/B1 scope uses locked worker assertions/hashes. No direct worker-memory audit, no per-query full-vocab matrices or raw gradients saved, no native utility inference.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    audit(parser.parse_args().seed)
