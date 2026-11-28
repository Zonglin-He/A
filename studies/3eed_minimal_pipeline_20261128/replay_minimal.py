"""Replay sealed queries, verify each equivalent pair, and audit the protected evidence."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from inspect import getsourcelines
from pathlib import Path
import argparse
import ast
import sys
import textwrap
from unittest.mock import patch

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '3eed_primary_formula_simplification_20261128'))
from ablate import ROOT, RUN as PREVIOUS, check, digest, fitting_audit, publish, read, sealed_inputs


RUN = ROOT / 'runs/minimal-pipeline-20261128'
BLIND_ARMS = ('Full', 'Minimal-A', 'Minimal-B', 'Minimal-C', 'C_single', 'C_dual')
EQUIVALENCE = 1.e-12


def variants(full):
    import minimal

    return {'Full': full, 'Minimal-A': partial(legacy, scales=minimal.SINGLE),
            'Minimal-B': partial(legacy, scales=minimal.DUAL), 'Minimal-C': minimal.predict,
            'C_single': partial(minimal.predict, scales=minimal.SINGLE),
            'C_dual': partial(minimal.predict, scales=minimal.DUAL)}


def legacy(frame, replies, *, scales):
    """Change the scale set at the one common legacy region selector."""
    import axiswise_2d as full
    import cray_simple

    choose = cray_simple.choose_region

    def at_scales(sample, terrain, _scales, fit_returns, in_window, selection=None):
        return choose(sample, terrain, scales, fit_returns, in_window, selection)

    with patch.object(full.live, 'choose_region', at_scales), patch.object(cray_simple, 'choose_region', at_scales):
        return full.predict(frame, replies)


def verify_output(old, new):
    assert old.keys() == new.keys()
    assert all(old[k] == new[k] for k in old if k != 'box'), (old, new)
    if old['box'] is None or new['box'] is None:
        assert old['box'] is new['box']
        return 0.
    delta = float(np.max(np.abs(np.asarray(old['box']) - new['box'])))
    assert delta <= EQUIVALENCE, (old, new, delta)
    return delta


def audit_view(detail):
    if detail is None:
        return dict(support_points=0, selection=0, audit=0, ties=False, candidates=[]), None

    from minimal import live
    evidence, candidates, ray_default = detail
    rays = evidence.selection, evidence.audit
    checks = []
    for box, reference, depth, protected in candidates:
        before = depth, live.first_depth(reference, evidence.audit, evidence.transform)
        checks.append(dict(fitting_audit(evidence, reference, box, protected, rays, before),
                           measured=len(protected[1]), entries=len(protected[2])))
    summary = dict(support_points=len(evidence.support_sensor), selection=len(evidence.selection),
                   audit=len(evidence.audit), ties=ray_default, candidates=checks)
    return summary, evidence


def verify_tie(detail):
    import axiswise_2d as full
    from minimal import select

    evidence, candidates, ray_default = detail
    if ray_default:
        cap = max(.1, .5 * min(evidence.reference_span[:2]))
        costs = [full.official.ray_score(reference, evidence.selection, evidence.transform, cap)
                 for _, reference, _, _ in candidates]
        winner = int(costs[1] < costs[0] - 1.e-12 * len(evidence.selection) * cap)
        assert select(evidence, candidates) == (winner, True)


def one(job):
    import axiswise_2d as full
    import minimal

    row, history, previous, originals = job
    frame, replies = row['frame'], row['raw']
    args = frame, replies
    variants_ = variants(full.predict)
    predictions = {arm: variants_[arm](*args) for arm in BLIND_ARMS[:3]}
    assert predictions['Full'] == history['outputs']['axis_2d']
    deltas, audits, bindings = {}, {}, {}
    for arm, base, scales in (('Minimal-C', 'Full', minimal.FOUR_SCALES),
                              ('C_single', 'Minimal-A', minimal.SINGLE), ('C_dual', 'Minimal-B', minimal.DUAL)):
        prediction, detail = minimal.predict(*args, scales=scales, audit=True)
        deltas[arm] = verify_output(predictions[base], prediction)
        predictions[arm] = prediction
        audits[arm], bindings[arm] = audit_view(detail)
        if detail is not None:
            verify_tie(detail)
    ref = bindings['Minimal-C']
    if ref is not None:
        origin = previous['audit']['formal']
        for candidate, old in zip(audits['Minimal-C']['candidates'], origin, strict=True):
            assert all(candidate[k] == v for k,v in old.items()), (candidate, old)
        if originals and originals['route'] == 'C_ray':
            rays = np.asarray(originals['selection']).reshape(-1, 3)
            held = np.asarray(originals['audit']).reshape(-1, 3)
        else:
            _, physical = full.live.incumbent(*args, collect_audit=True)
            assert physical is not None
            rays, held = physical[4:6]
        np.testing.assert_array_equal(rays, ref.selection)
        np.testing.assert_array_equal(held, ref.audit)
    else:
        assert not previous['audit']
    for arm, summary in audits.items():
        selected = bindings[arm]
        for name in ('selection', 'audit'):
            shared = set() if selected is None or ref is None else (
                {tuple(row) for row in getattr(ref, name)} & {tuple(row) for row in getattr(selected, name)})
            summary['overlap_' + name] = len(shared)
    audits.update(Full=audits['Minimal-C'], **{'Minimal-A': audits['C_single'], 'Minimal-B': audits['C_dual']})
    return dict(uid=frame['uid'], target=row['target'], predictions=predictions,
                audit=audits, equivalent_delta=deltas)


def toy_ranking():
    """Compare the exact ranking on valid, invalid, zero and tied projections."""
    from itertools import product
    from types import SimpleNamespace
    import minimal

    cases = 0
    for scores in product((None, 0., .2, .8), repeat=2):
        for ray in (0, 1):
            depths = ([5.], [9.]) if ray == 0 else ([9.], [5.])
            candidates = tuple((k, None, np.array(depth, float), None) for k, depth in enumerate(depths))
            evidence = SimpleNamespace(rectangle=(0,0,10,10), spec=None, transform=None, size=None,
                                       reference_span=(2,2,2), selection=np.array([[0.,0.,5.]]))
            with patch.object(minimal, 'projection_overlap', lambda box, *_: scores[box]):
                alt = scores[1 - ray]
                expected = 1 - ray if alt is not None and alt > (scores[ray] or 0.) else ray
                assert minimal.select(evidence, candidates) == (expected, (scores[0] or 0.) == (scores[1] or 0.))
            cases += 1
    return cases


def complexity():
    import axiswise_2d as full
    import cray_simple
    import minimal

    def count(functions):
        def body(fn):
            lines, _ = getsourcelines(fn)
            tree = ast.parse(textwrap.dedent(''.join(lines)))
            first = tree.body[0].body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                skipped = range(first.lineno, first.end_lineno + 1)
            else:
                skipped = ()
            return (sum(bool(line.strip()) and not line.lstrip().startswith('#') and i not in skipped
                        for i,line in enumerate(lines,1)),
                    sum(isinstance(node, (ast.If, ast.IfExp, ast.For, ast.While, ast.comprehension))
                        for node in ast.walk(tree)))
        sizes, branches = zip(*(body(f) for f in functions), strict=True)
        return dict(functions=len(functions), noncomment_source_lines=sum(sizes), control_nodes=sum(branches),
                    names=[f.__module__+'.'+f.__name__ for f in functions])

    return {'full': count((full.predict, full.prepare, full.official.prepare_pair, full.witnesses, full.fit,
                           full.live.clip_extent, full.rank, full.fallback, full.output,
                           cray_simple.predict, cray_simple.pose_box, cray_simple.choose_pose)),
            'unified': count((minimal.predict, minimal.bind, minimal.axis_shift, minimal.recover_pose, minimal.select))}


def run(limit):
    cv2.setNumThreads(1)
    def reject_labels(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            from os import fsdecode
            if Path(fsdecode(args[0])).name in {'scored-private.json', 'scores-private.json', 'test.parquet', 'data_object_label_2.zip'}:
                raise PermissionError('Offline labels are blocked during prediction')
    sys.addaudithook(reject_labels)
    source, originals, records, pins, _ = sealed_inputs()
    seal = check(PREVIOUS/'blind.json', 'payload', PREVIOUS/'prediction-seal.json')
    previous = {r['uid']: r for r in read(PREVIOUS/'blind.json')}
    toy = toy_ranking()
    ids = list(source)
    if limit:
        ids = [u for u in ids if records[u]['active']][:limit] + [u for u in ids if not records[u]['active']][:limit]
    tasks = ((source[u], records[u], previous[u], originals.get(u)) for u in ids)
    with ProcessPoolExecutor(8) as pool:
        predictions = []
        for prediction in pool.map(one, tasks, chunksize=3):
            predictions.append(prediction)
            if len(predictions) % 100 == 0 or len(predictions) == len(ids):
                print('Replayed', len(predictions), '/', len(ids), flush=True)
    print('Max box difference', {k:max(r['equivalent_delta'][k] for r in predictions)
                                  for k in predictions[0]['equivalent_delta']}, flush=True)
    if limit:
        return
    assert len(predictions) == 873
    RUN.mkdir(exist_ok=True, parents=True)
    out = RUN/'blind.json'
    publish(out, predictions)
    publish(RUN/'seal.json', dict(n=len(predictions), payload=digest(out), pins=pins, previous=seal['payload'],
            files={f:digest(HERE/f) for f in ('minimal.py','replay_minimal.py','PROTOCOL_ZH.md')},
            labels_read=False, model_calls=0, trained=False, toy_rank_cases=toy, complexity=complexity()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=int)
    run(parser.parse_args().smoke)
