"""Replay the locked R182 head on the reconstructed ordinary Validation roster."""

import argparse
from contextlib import nullcontext
from functools import lru_cache
from time import perf_counter
import sys
from unittest.mock import patch

import numpy as np
import torch

from scope183 import (ARMS, DIRECTIONS, HERE, INPUT, ROOT, TRAIN, PLATFORMS, atomic,
                      check_input, digest, frame, ordinary, read)

sys.path.insert(0, str(TRAIN))
from ranker182 import FeatureGrounder, K, SubjectRanker, readouts, R181
from fit182 import fitted
from capture181 import load_r180, write_npz, manifest as r181_manifest
from scope import evaluation_rows
from features import observation
from roles import parse_roles
from evaluate import to_sensor
from train import prepare_device
import readout181

OUT = HERE / 'predictions_PRIVATE'
BLOCK = 128
RUN_ORDER = tuple((s, t) for s in ('quad', 'waymo', 'drone')
                  for t in ({'quad': ('drone', 'waymo'), 'waymo': ('drone', 'quad'),
                             'drone': ('waymo', 'quad')}[s]))


def protocol():
    return read(HERE / 'PROTOCOL.json')


def ranker_weights(source):
    cfg = protocol()
    proof = cfg['frozen_equal_negative_heads'][source]
    lock = TRAIN / 'rankers_PRIVATE/FITTED.json'
    assert digest(lock) == cfg['r182_fit_receipt_sha256']
    entry = fitted()['sources'][source]['models']['generic']
    path = lock.parent / f'{source}_generic.pth'
    assert digest(path) == proof['sha256'] == entry['weights_sha256']
    assert entry['selected']['epoch'] == proof['epoch']
    content = torch.load(path, weights_only=True, map_location='cpu')
    assert content['name'] == 'generic' and content['source'] == source
    head = SubjectRanker()
    head.load_state_dict(content['state'], strict=True)
    return head.eval().requires_grad_(False)


def dependencies():
    paths = [HERE / n for n in ('PROTOCOL.json', 'scope183.py', 'predict183.py')]
    paths += [TRAIN / n for n in ('ranker182.py', 'fit182.py', 'PROTOCOL.json')]
    paths += [ROOT / 'crossplatform' / p for p in ('CURRENT_METHOD.json', 'candidate_frontend.py')]
    capture = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    capture.update(r181_manifest())
    return capture


@torch.inference_mode()
def ranking_fields(model, sample):
    """Match the R180 retriever inputs without materializing unused frozen features."""
    from grounding158 import select_grounding_box

    observed = {}
    scene, evidence = model._read_observation(sample, capture=observed)
    model._decoder(evidence)
    caption = sample['text'].strip()
    role = parse_roles(caption, observed['graph'])
    native = observed['native']
    role_tokens = role.mask(observed['offsets'], observed['valid'])
    content = (observed['offsets'][:, 1] > observed['offsets'][:, 0])
    content &= (observed['offsets'][:, 0] < len(caption)) & observed['valid']
    masks = torch.as_tensor(np.concatenate((role_tokens, content[None])),
                            device=native['xyz'].device).float()
    masks = masks / masks.sum(-1, keepdim=True).clamp_min(1.)
    token_proj = (masks @ native['tokens'][0]).half()
    _, utility = select_grounding_box(scene.boxes, scene.probability)
    prior = np.log(np.maximum(utility, 1e-100))
    prior = (prior - prior.max()).clip(-12, 0).astype(np.float32)
    # The frozen retriever reads only the three size columns, which PCA preserves.
    anchors = np.column_stack((scene.boxes, np.zeros(len(scene.boxes), dtype=np.float32)))
    anchors[:, 3:6] = anchors[:, 3:6].clip(min=1e-6)
    return dict(anchor_query=native['queries'][0].half(), token_proj=token_proj,
                anchors=anchors, prior=prior), role


class LockedEvaluator:
    def __init__(self, source):
        self.grounder = FeatureGrounder(source)
        self.head = ranker_weights(source)

    @torch.inference_mode()
    def evaluate(self, sample, debug=False):
        base = []

        def collect_four(_module, args):
            base.append(args[0]['subject_boxes'][0].cpu().numpy().copy())

        model = self.grounder
        with model._lock:
            hook = model._decoder.register_forward_pre_hook(collect_four)
            try:
                value = model.collect(sample)
            finally:
                hook.remove()
        quartet, = base
        assert quartet.shape == (4, 7)
        residual = self.head(torch.from_numpy(value['features'][None])).numpy()
        assert residual.shape == (1, K) and np.isfinite(residual).all()
        frozen = value['scores'][None]
        v1 = value['incumbent']
        r = readouts(residual, residual, frozen, np.array([v1]))
        choices = np.asarray((v1, *value['chosen'],
                              *(int(r[a][0]) for a in ARMS[4:])), dtype=np.int8)
        np.testing.assert_array_equal(value['boxes'][v1], quartet[v1])
        target, view = sample['target'], value['view']
        result = dict(choices=choices, boxes=to_sensor(value['boxes'], target, view),
                      four=to_sensor(quartet, target, view),
                      outside=~np.isin(value['slate'], value['base_slots']),
                      active=bool(value['has_reference']),
                      max_residual=float(np.abs(residual).max()))
        assert not result['outside'][v1]
        if debug:
            result.update(slate=value['slate'], base_slots=value['base_slots'],
                          scores=value['scores'], residual=residual, features=value['features'])
        return result


def parity():
    """Check stable old predictions and bit equality of both ordinary inference paths."""
    rows = evaluation_rows()
    at = ordinary()
    old = load_r180(rows)
    proof = read(TRAIN / 'predictions_PRIVATE/SEALED.json')
    assert proof['n'] == 3072 and not proof['FINAL_access']
    known = truthless = 0
    timing = dict(ordinary_seconds=0., optimized_seconds=0.)
    fast = False
    for source in PLATFORMS:
        obj = LockedEvaluator(source)
        parser = lru_cache(maxsize=32)(obj.grounder._frontend.parser)
        for s, t in DIRECTIONS:
            if s != source:
                continue
            block = old[s, t]
            path = TRAIN / 'predictions_PRIVATE' / f'{s}_to_{t}.npz'
            assert digest(path) == proof['files'][path.name]['sha256']
            with np.load(path, allow_pickle=False) as store:
                legacy = store['chosen']
            for i in (0, 257):
                sample = observation(rows[int(block['ordinals'][i])])
                observed = obj.evaluate(sample, debug=True)
                ideal = block['slate'][i, 6, :K]
                np.testing.assert_array_equal(observed['slate'], ideal)
                np.testing.assert_array_equal(observed['boxes'], block['pool'][i, ideal])
                np.testing.assert_array_equal(observed['choices'][:4], legacy[i, :4])
                assert observed['choices'][ARMS.index('rank_joint')] == legacy[i, 7]
                known += 1
            originals = obj.grounder._frontend.parser
            for j in (0, len(at[t]) // 2, len(at[t]) - 1):
                example = at[t][j]
                check_input(example)
                sample = frame(example)
                tick = perf_counter()
                baseline = obj.evaluate(sample, debug=True)
                timing['ordinary_seconds'] += perf_counter() - tick
                obj.grounder._frontend.parser = parser
                try:
                    with patch.object(readout181, 'encode', ranking_fields):
                        tick = perf_counter()
                        accelerated = obj.evaluate(sample, debug=True)
                        timing['optimized_seconds'] += perf_counter() - tick
                finally:
                    obj.grounder._frontend.parser = originals
                for key in baseline:
                    np.testing.assert_array_equal(baseline[key], accelerated[key], err_msg=key)
                truthless += 1
        del obj, parser
        torch.cuda.empty_cache()
    assert known == 12 and truthless == 18
    fast = True
    atomic(HERE / 'PARITY.json', dict(old_TRAIN_rows=known, ordinary_Validation_rows=truthless,
           generic_joint_matches_R182=known, all_selected_boxes_and_features_bit_exact=truthless,
           implementation=dependencies(), optimization_allowed=fast, target_labels_read=False,
           FINAL_access=False, **timing))
    print('Historical and ordinary parity passed', known, truthless, timing, flush=True)


def execution():
    check = read(HERE / 'PARITY.json')
    assert check['implementation'] == dependencies() and not check['target_labels_read']
    assert check['generic_joint_matches_R182'] == 12 and not check['FINAL_access']
    assert check['all_selected_boxes_and_features_bit_exact'] == 18
    cfg = protocol()
    assert tuple(cfg['arms']) == ARMS and cfg['ordinary_cohort']['direction_records'] == 18620
    assert all(not cfg[k] for k in ('source_training_runs', 'target_updates', 'FINAL_access', 'hyperparameter_search'))
    return dict(implementation=dependencies(), roster=cfg['input_roster_seal_sha256'],
                parity=digest(HERE / 'PARITY.json'), block_size=BLOCK, auto_mode='exact_rank_fields',
                arms=list(ARMS), target_labels_read=False, FINAL_access=False)


def run():
    targets = ordinary()
    lock = execution()
    OUT.mkdir(mode=0o700, exist_ok=True)
    p = OUT / 'LOCK.json'
    if p.exists():
        assert read(p) == lock
    else:
        atomic(p, lock)
    assert not (OUT / 'SEALED.json').exists()
    start = perf_counter()
    completed = {}
    parse_cache = None
    with patch.object(readout181, 'encode', ranking_fields):
        for source in ('quad', 'waymo', 'drone'):
            obj = LockedEvaluator(source)
            if parse_cache is None:
                parse_cache = lru_cache(maxsize=12000)(obj.grounder._frontend.parser)
            obj.grounder._frontend.parser = parse_cache
            for s, t in RUN_ORDER:
                if s != source:
                    continue
                direction = f'{s}_to_{t}'
                part = OUT / direction
                part.mkdir(mode=0o700, exist_ok=True)
                rows = targets[t]
                ch = {}
                for left in range(0, len(rows), BLOCK):
                    right = min(left + BLOCK, len(rows))
                    path = part / f'{left:05d}.npz'
                    pin = path.with_suffix('.json')
                    if path.exists() and pin.exists():
                        receipt = read(pin)
                        assert receipt == dict(first=left, last=right, n=right-left, sha256=digest(path))
                    else:
                        assert not pin.exists()
                        if path.exists():
                            path.replace(path.with_suffix('.unsealed'))
                        values = []
                        for row in rows[left:right]:
                            check_input(row)
                            values.append(obj.evaluate(frame(row)))
                        arrays = {k: np.stack([v[k] for v in values]) for k in values[0]}
                        assert arrays['boxes'].shape == (right-left, K, 7)
                        assert arrays['choices'].shape == (right-left, len(ARMS))
                        assert np.isfinite(arrays['boxes']).all() and (arrays['boxes'][..., 3:6] > 0).all()
                        assert np.all((0 <= arrays['choices']) & (arrays['choices'] < K))
                        write_npz(path, dict(ordinals=np.arange(left, right), **arrays))
                        receipt = dict(first=left, last=right, n=right-left, sha256=digest(path))
                        atomic(pin, receipt)
                    ch[path.name] = receipt
                    print(direction, right, len(rows), round(perf_counter() - start, 1),
                          'cache', parse_cache.cache_info().hits, flush=True)
                completed[direction] = dict(n=len(rows), files=ch)
            del obj
            torch.cuda.empty_cache()
    assert set(completed) == {f'{s}_to_{t}' for s, t in DIRECTIONS}
    assert sum(group['n'] for group in completed.values()) == 18620
    atomic(OUT / 'SEALED.json', dict(n=18620, directions=completed, lock_sha256=digest(p),
                                    roster_sha256=digest(INPUT / 'SEALED.json'), all_arms_same_slate=True,
                                    source_updates=0, target_updates=0, target_labels_read=False,
                                    FINAL_access=False, elapsed_seconds=perf_counter() - start))
    print('Sealed full-matrix predictions', 18620, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('parity', 'run'))
    mode = parser.parse_args().phase
    prepare_device()
    {'parity': parity, 'run': run}[mode]()
