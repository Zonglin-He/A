"""Replay fixed source evidence, then seal the ordinary and predicate-reversed decisions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gc
import multiprocessing
import os
import time

import numpy as np

from scope167 import ARMS, HERE, ORDER, PLATFORMS, PROTOCOL, ROOT, W165, W166, WORK, atomic_json, lock, parents, read, roster, sha, status
from retrieval167 import blank, direct_edges, repaired
from predict166 import capture_case
from evidence162 import EvidenceGrounder


def save_block(path, items, ordinals, seals):
    tmp = path.with_suffix('.tmp')
    fields = {name: np.stack([row[name] for row in items]) for name in items[0]}
    with tmp.open('wb') as f:
        np.savez_compressed(f, **fields, ordinals=ordinals, arms=np.asarray(ARMS),
                            parents=np.asarray(seals), protocol=sha(PROTOCOL))
    os.replace(tmp, path)


def inspect_parent(target, a, b, k):
    ordinal = int(a['ordinals'][k])
    row = roster()[target][ordinal]
    out = {name: b[name][k] for name in ('starts', 'budget', 'routes')}
    out['edges'] = b['edges'][b['edge_offsets'][k]:b['edge_offsets'][k + 1]]
    baseline = a['boxes'][k, -1]
    np.testing.assert_array_equal(b['boxes'][k, 0, 0], baseline)
    np.testing.assert_array_equal(a['ordinals'][k], b['ordinals'][k])
    winner = int(a['winners'][k, -1])
    return ordinal, row, out, baseline, winner


def pilot_ordinals(a, b):
    seen = {0: 0, 1: 0, 2: 0}
    selected = []
    for key in sorted(a):
        with np.load(b.parent / key, allow_pickle=False) as f:
            edges, off = f['edges'], f['edge_offsets']
            for ordinal, lo, hi in zip(f['ordinals'], off[:-1], off[1:], strict=True):
                length = min(2, len(direct_edges(edges[lo:hi])))
                if seen[length] < (2 if length == 0 else 3):
                    seen[length] += 1
                    selected.append(int(ordinal))
        if sum(seen.values()) == 8:
            break
    assert sum(seen.values()) == 8
    return set(selected)


def predict_direction(model, source, target, pool, pilot, start):
    name = f'{source}_to_{target}'
    source_dir = W165 / 'ordinary' / name
    previous_dir = W166 / 'ordinary' / name
    a, b = [read(p / 'SEAL.json') for p in (source_dir, previous_dir)]
    assert a['n'] == b['n'] == len(roster()[target]) and set(a['files']) == set(b['files'])
    keep = pilot_ordinals(a['files'], previous_dir / 'SEAL.json') if pilot else None
    dest = WORK / ('pilot' if pilot else 'ordinary') / name
    dest.mkdir(parents=True, exist_ok=True)
    files, total, captures = {}, 0, 0
    for key, h0 in sorted(a['files'].items()):
        h1 = b['files'][key]
        raw_file, bank_file = source_dir / key, previous_dir / key
        assert sha(raw_file) == h0 and sha(bank_file) == h1
        with np.load(raw_file, allow_pickle=False) as z, np.load(bank_file, allow_pickle=False) as y:
            # NPZ member reads decompress entire arrays. Load once per block.
            raw, bank = dict(z), dict(y)
        assert str(bank['parent_block_sha256']) == h0
        indexes = [k for k, n in enumerate(raw['ordinals']) if keep is None or int(n) in keep]
        if not indexes:
            continue
        ordinals = raw['ordinals'][indexes]
        out = dest / key
        if not out.exists():
            work = []
            for k in indexes:
                _, row, old, baseline, winner = inspect_parent(target, raw, bank, k)
                if not direct_edges(old['edges']):
                    work.append(blank(baseline, winner))
                    continue
                ev, _ = capture_case(model, row, baseline, raw['raw'][k], raw['points_sha'][k],
                                     raw['indices_sha'][k], pilot and captures == 0)
                captures += 1
                np.testing.assert_array_equal(bank['seeds'][k], ev.seeds)
                np.testing.assert_array_equal(old['edges'], ev.edges)
                lo, hi = raw['offsets'][k:k + 2]
                work.append(pool.submit(repaired, ev, old, raw['q'][k], raw['tokens'][lo:hi]))
            values = [r if isinstance(r, dict) else r.result() for r in work]
            save_block(out, values, ordinals, (h0, h1))
        else:
            with np.load(out, allow_pickle=False) as check:
                assert check['parents'].tolist() == [h0, h1] and str(check['protocol']) == sha(PROTOCOL)
                np.testing.assert_array_equal(ordinals, check['ordinals'])
        files[key] = sha(out)
        total += len(ordinals)
        if pilot or total % 512 == 0 or total == a['n']:
            print('R167', name, total, '/', 8 if pilot else a['n'], 'seconds', round(time.perf_counter() - start, 1), flush=True)
    assert total == (8 if pilot else a['n'])
    value = dict(n=total, files=files, protocol=sha(PROTOCOL), parent_native=sha(source_dir / 'SEAL.json'),
                 parent_lcib=sha(previous_dir / 'SEAL.json'), target_label_read=False, FINAL_access=False)
    atomic_json(dest / 'SEAL.json', value)
    return dict(n=total, seal=sha(dest / 'SEAL.json'))


def run(pilot):
    import torch
    from model51 import fingerprint

    lock()
    parents()
    if not pilot:
        r = read(HERE / 'PILOT_SEAL.json')
        assert r['n'] == 48 and r['input_lock'] == sha(HERE / 'INPUT_LOCK.json')
    status('preflight' if pilot else 'running')
    start = time.perf_counter()
    direction_seals = {}
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        with ProcessPoolExecutor(max_workers=5, mp_context=multiprocessing.get_context('spawn')) as pool:
            for source in PLATFORMS:
                model = EvidenceGrounder(source)
                state = (fingerprint(model.base), fingerprint(model.matcher))
                for a, target in ORDER:
                    if a == source:
                        name = f'{source}_to_{target}'
                        direction_seals[name] = predict_direction(model, source, target, pool, pilot, start)
                assert state == (fingerprint(model.base), fingerprint(model.matcher))
                del model
                gc.collect()
                torch.cuda.empty_cache()
    atomic_json(HERE / ('PILOT_SEAL.json' if pilot else 'PREDICTION_SEAL.json'),
                dict(n=sum(x['n'] for x in direction_seals.values()), directions=direction_seals,
                     input_lock=sha(HERE / 'INPUT_LOCK.json'), FINAL_access=False, target_label_read=False,
                     source_training_runs=0, unchanged_model_fingerprints=True,
                     seconds=time.perf_counter() - start))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    run(p.parse_args().pilot)
