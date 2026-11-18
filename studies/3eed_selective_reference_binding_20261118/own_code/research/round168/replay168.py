"""Replay R167 cached assignments and freeze all new predictions before target labels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
from functools import lru_cache
import gc
import hashlib
from multiprocessing import get_context
import os

import numpy as np
from scipy.special import expit

from scope168 import ARMS, HERE, ORDER, PLATFORMS, R167, ROOT, SENTINEL, TOK, WORK, atomic_json, lock, read, roster, sha, status
from audit167 import blocks, row_edges
from common160 import array_sha
from labels166 import parser_record
from predict165 import points_for
from routing88 import token_roles88
from select168 import legacy_availability, policies, role_terms
from crossplatform.ground_single_frame import orient_box


def rule():
    p = read(HERE / 'ABSTENTION_RULE.json')
    assert not p['target_label_used'] and not p['FINAL_access']
    assert sha(HERE / 'SOURCE_FEATURE_SEAL.json') == p['source_feature_sha256']
    return p['rules']


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import RobertaTokenizerFast
    return RobertaTokenizerFast.from_pretrained(str(TOK), local_files_only=True)


def word_masks(row, document, length, word_tokenizer=None):
    from types import SimpleNamespace

    tok = word_tokenizer or tokenizer()
    obj = SimpleNamespace(tokenizer=tok)
    masks, _ = token_roles88(obj, row, {'lang0': np.empty((1, length, 0))}, document['graph'], document['doc'])
    text = row['caption'] + SENTINEL
    valid = np.asarray(tok(text)['attention_mask'], bool)
    assert len(valid) == length and valid.all()
    return masks, valid


def save(path, records, **extra):
    arrays = {k: np.stack([x[k] for x in records]) for k in records[0]}
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as s:
        np.savez_compressed(s, **arrays, **extra)
    os.replace(tmp, path)


def oriented(row, old, native, choices, get_pts):
    memo = {-1: np.full(7, np.nan)}
    # R167 wrote the exact geometry of every selected candidate in both signs.
    for k in ('', 'flip_'):
        for j, v in zip(old[k + 'winners'].flat, old[k + 'boxes'].reshape(-1, 7), strict=True):
            if j not in memo:
                memo[j] = v
            elif j >= 0:
                np.testing.assert_array_equal(memo[j], v)
    n = len(native)
    points = None
    for q in np.unique(choices):
        if q in memo:
            continue
        if q >= n:
            memo[q] = old['novel'][q - n]
        else:
            if points is None:
                points = get_pts(row)
            memo[q] = orient_box(points, native[q])
    assembled = np.asarray([memo[j] for j in choices.flat]).reshape(*choices.shape, 7)
    assert np.isfinite(assembled[choices >= 0]).all()
    return assembled


def infer(row, old, raw, seeds, q, t, roles, valid, edges, params, key, get_pts):
    common = tuple(old[x] for x in ('bank', 'score', 'left', 'overlap'))
    bank, old_score, left, overlap = common
    if edges:
        new, delta, feat = role_terms(raw, old['novel'], seeds, q, t, roles, valid, bank, old_score)
    else:
        new, delta = old_score, np.zeros_like(old_score)
        feat = np.full((len(bank), 2), np.nan)
    pred, contribution, logits, changed = policies(*common, edges, new, delta, feat, params, key)
    np.testing.assert_array_equal(pred[:2], old['winners'][[1, 3]])
    if old['flip']:
        cf, cf_effect, _, _ = policies(*common, edges, new, delta, feat, params, key, swap=True)
        np.testing.assert_array_equal(cf[:2], old['flip_winners'][[1, 3]])
    else:
        cf, cf_effect = pred, contribution
    geometry = oriented(row, old, raw, np.stack((pred, cf)), get_pts)
    np.testing.assert_array_equal(geometry[0, :2], old['boxes'][[1, 3]])
    np.testing.assert_array_equal(geometry[1, :2], old['flip_boxes'][[1, 3]])
    # Scores relative to the independent subject are private per candidate, while
    # the resulting scalar component differences may be published.
    return dict(chosen=pred, flip_chosen=cf, boxes=geometry[0], flip_boxes=geometry[1],
                decomposition=contribution, flip_decomposition=cf_effect, new_score=new,
                gate_p=expit(logits), cf=old['flip'], legacy_advantage=np.min(legacy_availability(*common, edges), initial=np.inf),
                n_physical_roles=int((bank[1:] >= 0).any(1).sum()), effective_permutations=changed)


def direction(source, target, stage):
    tag = f'{source}_to_{target}'
    dest = WORK / stage / tag
    dest.mkdir(exist_ok=True, parents=True)
    params = rule()[source]
    known = read(R167 / ('PILOT_SEAL.json' if stage == 'pilot' else 'PREDICTION_SEAL.json'))
    outputs = {}
    results = Counter()
    shown = ('bank', 'score', 'left', 'overlap', 'winners', 'flip_winners', 'boxes', 'flip_boxes', 'novel', 'flip')
    for name, pred, base, earlier in blocks(stage, source, target):
        work = []
        for i, j in enumerate(pred['ordinals']):
            k = int(np.searchsorted(base['ordinals'], j))
            row = roster()[target][int(j)]
            old = {field: pred[field][i] for field in shown}
            _, direct = row_edges(earlier, k)
            raw = base['raw'][k]
            if direct:
                lo, hi = base['offsets'][k:k+2]
                t, q = base['tokens'][lo:hi], base['q'][k]
                parse = parser_record(row['id'])
                assert hashlib.sha256(row['caption'].encode()).hexdigest() == parse['caption_sha256']
                roles, valid = word_masks(row, parse, len(t))
                np.testing.assert_array_equal(earlier['spans'][k, :len(roles)],
                                              np.array([e['span'] for e in parse['graph']['entities']]))
            else:
                q = t = roles = valid = None

            def get_pts(row):
                pts, ix = points_for(row)
                assert bytes(base['points_sha'][k]) == bytes.fromhex(array_sha(pts))
                assert bytes(base['indices_sha'][k]) == bytes.fromhex(array_sha(ix))
                return pts

            out = infer(row, old, raw, earlier['seeds'][k], q, t, roles, valid, direct, params, row['id'], get_pts)
            work.append(out)
            results.update(n=1, direct=int(bool(direct)), flip=int(old['flip']),
                           physical_roles=out['n_physical_roles'], permutations=out['effective_permutations'])
            totals = np.isfinite(out['gate_p']).sum()
            assert totals == out['n_physical_roles']
            assert out['legacy_advantage'] > 0
        out_path = dest / name
        save(out_path, work, ordinals=pred['ordinals'], arms=ARMS)
        outputs[name] = sha(out_path)
        if results['n'] % 1024 == 0:
            print('R168 replay', tag, results['n'], flush=True)
    assert results['n'] == known['directions'][tag]['n']
    sealed = dict(n=results['n'], counts=dict(results), files=outputs, parent_seal=known['directions'][tag]['seal'],
                  target_label_read=False, FINAL_access=False)
    atomic_json(dest / 'SEAL.json', sealed)
    print('R168 sealed', stage, tag, sealed['counts'], flush=True)
    return dict(seal=sha(dest / 'SEAL.json'), **results)


def run(stage):
    lock()
    assert not read(HERE / 'ABSTENTION_RULE.json')['target_label_used']
    if stage == 'ordinary':
        assert read(HERE / 'PILOT_SEAL.json')['n'] == 48
        status('replay')
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as pool:
        future = {(s, t): pool.submit(direction, s, t, stage) for s, t in ORDER}
        seals = {f'{s}_to_{t}': future[s, t].result() for s, t in ORDER}
    n = sum(d['n'] for d in seals.values())
    assert n == (48 if stage == 'pilot' else 18620)
    name = 'PILOT_SEAL.json' if stage == 'pilot' else 'PREDICTION_SEAL.json'
    atomic_json(HERE / name, dict(n=n, directions=seals, rule=sha(HERE / 'ABSTENTION_RULE.json'),
                input_lock=sha(HERE / 'INPUT_LOCK.json'), exact_r167_replay=True, target_label_read=False, FINAL_access=False))


def controlled():
    import torch
    from challenge167 import DIR as R167CTL, evidence, parse
    from review167 import REGISTRY, locate
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint
    from retrieval167 import repaired

    lock()
    init = read(R167 / 'CONTROLLED_PREDICTION_SEAL.json')
    reg = {r['case']: r for r in read(REGISTRY)}
    assert sha(R167CTL / 'QUERIES.json') == read(R167 / 'CONTROLLED_LOCK.json')['clauses']
    cases = read(R167CTL / 'QUERIES.json')
    dest = WORK / 'controlled'
    dest.mkdir(exist_ok=True, parents=True)
    saved = {}
    rules = rule()
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as g:
        fcntl.flock(g, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for source in PLATFORMS:
            m = EvidenceGrounder(source)
            initial = fingerprint(m.base), fingerprint(m.matcher)
            for case in cases:
                case_id = case['case']
                target = reg[case_id]['target']
                if source == target:
                    continue
                ordinal = reg[case_id]['ordinal']
                row = roster()[target][ordinal]
                a, b, sign = parse(case['left'])
                k, _, older = locate(source, target, ordinal)
                with torch.inference_mode():
                    ev, q, t, route = evidence(m, row, a, b)
                prior = {x: older[x][k] for x in ('starts', 'budget')}
                prior['routes'] = route
                latest = repaired(ev, prior, q, t)
                filename = f'{case_id:02d}_{source}_to_{target}.npz'
                assert sha(R167CTL / filename) == init['files'][filename]
                with np.load(R167CTL / filename, allow_pickle=False) as f:
                    for x in ('bank', 'score', 'boxes', 'flip_boxes', 'winners', 'flip_winners', 'novel', 'left', 'overlap'):
                        np.testing.assert_array_equal(latest[x], f[x])
                    np.testing.assert_array_equal(ev.seeds, f['seeds'])
                record = infer(row, latest, ev.boxes, ev.seeds, q, t, ev.role_mask, np.ones(len(t), bool),
                               [(1, sign)], rules[source], f'controlled|{case_id}', lambda _: ev.points)
                path = dest / filename
                save(path, [record], seeds=ev.seeds, arms=ARMS)
                saved[filename] = sha(path)
            assert initial == (fingerprint(m.base), fingerprint(m.matcher))
            del m
            gc.collect()
            torch.cuda.empty_cache()
    assert len(saved) == 10
    atomic_json(HERE / 'CONTROLLED_PREDICTION_SEAL.json', dict(n=10, files=saved,
                rule=sha(HERE / 'ABSTENTION_RULE.json'), target_label_read=False, FINAL_access=False))
    print('R168 controlled pairs sealed', len(saved), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pilot', 'ordinary', 'controlled'))
    stage = parser.parse_args().stage
    controlled() if stage == 'controlled' else run(stage)
