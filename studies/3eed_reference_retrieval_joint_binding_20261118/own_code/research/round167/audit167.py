"""Independent checks of the common bank and the paired predicate intervention."""
import argparse
from collections import Counter
from collections.abc import Iterator
from itertools import product

import numpy as np

from scope167 import ARMS, HERE, NROLE, ORDER, PROTOCOL, W165, W166, WORK, atomic_json, lock, read, sha


def blocks(cohort, source, target, only_file=None) -> Iterator[tuple[str, dict, dict, dict]]:
    path = HERE / ('PILOT_SEAL.json' if cohort == 'pilot' else 'PREDICTION_SEAL.json')
    proof = read(path)
    assert proof['input_lock'] == sha(HERE / 'INPUT_LOCK.json') and not proof['FINAL_access']
    assert not proof['target_label_read'] and proof['unchanged_model_fingerprints']
    direction = f'{source}_to_{target}'
    dest = WORK / cohort / direction
    forward = W165 / 'ordinary' / direction
    lcib = W166 / 'ordinary' / direction
    digests = [sha(x / 'SEAL.json') for x in (dest, forward, lcib)]
    run, old, init = [read(x / 'SEAL.json') for x in (dest, forward, lcib)]
    assert digests == [proof['directions'][direction]['seal'], run['parent_native'], run['parent_lcib']]
    entries = [(only_file, run['files'][only_file])] if only_file else sorted(run['files'].items())
    for name, digest in entries:
        files = [x / name for x in (dest, forward, lcib)]
        hashes = [digest, old['files'][name], init['files'][name]]
        assert [sha(x) for x in files] == hashes
        with np.load(files[0], allow_pickle=False) as a, np.load(files[1], allow_pickle=False) as b, np.load(files[2], allow_pickle=False) as c:
            assert tuple(a['arms']) == ARMS and a['parents'].tolist() == hashes[1:]
            assert str(a['protocol']) == sha(PROTOCOL)
            pred, native, earlier = dict(a), dict(b), dict(c)
        positions = np.searchsorted(native['ordinals'], pred['ordinals'])
        np.testing.assert_array_equal(native['ordinals'], earlier['ordinals'])
        np.testing.assert_array_equal(pred['ordinals'], native['ordinals'][positions])
        np.testing.assert_array_equal(pred['boxes'][:, 0, 0], native['boxes'][positions, -1])
        yield name, pred, native, earlier


def row_edges(record, index):
    lo, hi = record['edge_offsets'][index:index + 2]
    edges = record['edges'][lo:hi]
    direct = list(dict.fromkeys((int(b), int(p)) if a == 0 else (int(a), -int(p))
                               for a, b, p in edges if (a == 0) != (b == 0)))
    return edges, direct


def check_block(pred, native, previous, brute=False):
    checks = Counter()
    for k, loc in enumerate(np.searchsorted(previous['ordinals'], pred['ordinals'])):
        _, direct = row_edges(previous, loc)
        expected_flip = len(direct) == 1
        assert expected_flip == pred['flip'][k]
        base = pred['boxes'][k, 0, 0]
        if not direct:
            np.testing.assert_array_equal(pred['boxes'][k, :, 0], np.repeat(base[None], len(ARMS), axis=0))
            np.testing.assert_array_equal(pred['flip_winners'][k], pred['winners'][k])
            checks['R158_fallback'] += 1
            continue
        checks['direct_records'] += 1
        checks['predicted_flip_pairs'] += int(expected_flip)
        bank = [list(row[row >= -1]) for row in pred['bank'][k]]
        scores = [list(row[:len(b)]) for row, b in zip(pred['score'][k], bank, strict=True)]
        roles = sorted({r for r, _ in direct})
        for role in [0, *roles]:
            assert len(set(bank[role])) == len(bank[role])
            assert np.isfinite(scores[role]).all()
            if role:
                assert bank[role][-1] == -1
        n = len(bank[0])
        for swap in (False, True) if expected_flip else (False,):
            found = pred['flip_winners' if swap else 'winners'][k]
            terms = {}
            for role in roles:
                width = len(bank[role])
                sign = (-1 if swap else 1) * sum(p for r, p in direct if r == role)
                left = pred['left'][k, role - 1, :n, :width]
                cost = pred['overlap'][k, role - 1, :n, :width]
                terms[role] = np.asarray(scores[role])[None, :] + sign * left - cost
                assert np.all(np.abs(left) <= 1) and np.all((cost >= 0) & (cost <= 1 + 1.e-10))
            independently_selected = [max(range(len(v)), key=v.__getitem__) if v else None for v in scores]
            fixed = {r: independently_selected[r] for r in roles}
            values = (scores[0], [scores[0][i] + sum(terms[r][i, fixed[r]] for r in roles) for i in range(n)],
                      [scores[0][i] + sum(max(terms[r][i]) for r in roles) for i in range(n)])
            for arm, v in enumerate(values, 1):
                # Ties may differ at floating-point roundoff. Verify the score and the selected endpoint logic.
                i = bank[0].index(int(found[arm, 0]))
                assert abs(v[i] - max(v)) < 1.e-10
                for r in roles:
                    j = bank[r].index(int(found[arm, r]))
                    if arm < 3:
                        assert j == fixed[r]
                    else:
                        assert abs(terms[r][i, j] - max(terms[r][i])) < 1.e-10
                checks['matched_assignments'] += 1
            if brute and n * np.prod([len(bank[r]) for r in roles]) <= 50000:
                best = max(scores[0][i] + sum(terms[r][i, j] for r, j in zip(roles, js, strict=True))
                           for i, *js in product(range(n), *(range(len(bank[r])) for r in roles)))
                assert abs(best - max(values[-1])) < 1.e-10
                checks['exhaustive_joint_objectives'] += 1
        if not expected_flip:
            np.testing.assert_array_equal(pred['flip_winners'][k], pred['winners'][k])
        np.testing.assert_array_equal(pred['boxes'][k, 1, 0], base)
        checks['R158_independent_subject_exact'] += 1
    return checks


def audit(cohort):
    lock()
    total, n = Counter(), 0
    for source, target in ORDER:
        for _, pred, native, old in blocks(cohort, source, target):
            total.update(check_block(pred, native, old, brute=cohort == 'pilot'))
            n += len(pred['ordinals'])
    assert n == (48 if cohort == 'pilot' else 18620)
    audit = dict(n=n, counts=dict(total), FINAL_access=False, target_label_read=False)
    atomic_json(HERE / ('PILOT_AUDIT.json' if cohort == 'pilot' else 'READOUT_AUDIT.json'), audit)
    print(cohort, n, dict(total), flush=True)
    return audit


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('cohort', choices=('pilot', 'ordinary'))
    audit(arg.parse_args().cohort)
