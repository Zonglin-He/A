"""Replay matched, label-free geometry with a single change of surface."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
OABC = HERE.parent / '3eed_oabc_pose_center_feasibility_20261121'
PRIVATE, OUT, SPEC = HERE / 'private', HERE / 'results', HERE / 'PROTOCOL.json'
SOURCE_SEAL = OABC / 'results/prediction_seal.json'
SOURCE_PREDICTIONS = OABC / 'private/predictions.json'
sys.path.insert(0, str(OABC))

from feasible_geometry import CameraRays, continuous_fit, rotation
from oabc import (BASE, POSES, PREVIOUS, camera_evidence, center_box, digest,
                  evenly_spaced, read, to_raw_support, transfer, verify_sources, write)
from shape_proxy import QuarticRays


ARMS = ('A', 'B', 'C')


def read_source():
    seal = read(SOURCE_SEAL)
    assert digest(SOURCE_PREDICTIONS) == seal['predictions']
    assert digest(OABC / 'PROTOCOL.json') == seal['protocol']
    assert digest(PREVIOUS / 'results/prediction_seal.json') == seal['previous_prediction_seal']
    assert digest(BASE / 'private/panel.json') == seal['panel']
    assert all(digest(OABC / name) == value for name, value in seal['implementation'].items())
    assert not seal['target_labels_read'] and not seal['FINAL_access']
    assert seal['ray_isolation']['retained_withheld_raw'] == 0
    return seal, read(SOURCE_PREDICTIONS)


def select_pose(family, candidates, rays, transform, cap):
    scores = {}
    for pose, box in candidates.items():
        axes = rotation(box[6])
        view = family(rays, transform, box[6], box[3:6], cap)
        scores[pose] = float(view.losses(box[:3] @ axes)[0])
    return min(scores, key=scores.__getitem__), scores


def replay_one(args):
    frame, replies, previous, _, cfg, source = args
    boxes = source['boxes']
    estimate = dict(boxes=dict(A=boxes['A'], B=boxes['plain_pair'], C=boxes['anchor_pair']),
                    fit_node=source['fit_node'], used=False, status=source['status'])
    if not source['fit_node']:
        estimate['status'] = 'no_blind_region'
        assert estimate['boxes']['B'] == estimate['boxes']['C']
        return estimate
    if len(source['fit_rays']) < cfg['minimum_rays'] or source['selection_count'] < cfg['minimum_rays']:
        estimate['status'] = 'insufficient_fit_or_selection'
        assert estimate['boxes']['B'] == estimate['boxes']['C']
        return estimate

    context = camera_evidence(frame, replies, previous)
    cuts, labels, births, _ = context['catalog']
    node = context['node']
    np.testing.assert_array_equal(to_raw_support(node['points'], frame['target'], frame['suffix']),
                                  source['support'])
    membership = transfer(context['fit'], context['points'], cuts, labels, births)[node['level']] == node['node']
    fit, select, audit = [context['rays'][membership & (context['fold'] == fold)] for fold in range(3)]
    fit = evenly_spaced(fit, cfg['fit_ray_cap'])
    np.testing.assert_array_equal(fit, source['fit_rays'])
    np.testing.assert_array_equal(audit, np.asarray(source['audit']).reshape(-1, 3))
    assert len(select) == source['selection_count'] and context['isolation'] == source['isolation']
    transform = np.asarray(source['transform'])
    np.testing.assert_array_equal(transform, context['transform'])

    dims = np.asarray(source['dims'])
    cap = max(.1, .5 * min(dims[:2]))
    plain, proxy, fit_loss = {}, {}, {}
    for pose in POSES:
        old = source['poses'][pose]
        anchor = np.asarray(old['anchor'])
        yaw = anchor[6]
        np.testing.assert_array_equal(dims, anchor[3:6])
        initial = anchor[:3] @ rotation(yaw)
        domain_low, domain_high = np.asarray(old['domain'])
        low = np.minimum(initial - dims, domain_low)
        high = np.maximum(initial + dims, domain_high)
        rays = QuarticRays(fit, transform, yaw, dims, cap)
        center, loss, _, _ = continuous_fit(rays, low, high, initial,
            cfg['grid_samples_per_axis'], cfg['powell_evaluations_per_start'])
        plain[pose] = np.asarray(old['plain'])
        proxy[pose] = np.asarray(center_box(center, yaw, dims))
        fit_loss[pose] = {'B': old['plain_fit_loss'], 'C': loss}
    box_pose, old_cost = select_pose(CameraRays, plain, select, transform, cap)
    old_pick = source['choices']['plain_pair']
    assert box_pose == old_pick['pose']
    np.testing.assert_allclose(list(old_cost.values()), [old_pick['selection_losses'][p] for p in POSES],
                               rtol=0., atol=1.e-12)
    np.testing.assert_array_equal(plain[box_pose], estimate['boxes']['B'])
    shape_pose, shape_cost = select_pose(QuarticRays, proxy, select, transform, cap)
    estimate['boxes']['C'] = proxy[shape_pose].tolist()
    estimate.update(used=True, status='fitted', fit_count=len(fit), selection_count=len(select),
                    audit_count=len(audit), box_pose=box_pose, shape_pose=shape_pose,
                    box_pose_proxy=proxy[box_pose].tolist(),
                    fit_loss={'B': fit_loss[box_pose]['B'], 'C': fit_loss[shape_pose]['C']},
                    select_loss={'B': old_cost[box_pose], 'C': shape_cost[shape_pose]})
    return estimate


def run():
    started = perf_counter()
    assert not (OUT / 'prediction_seal.json').exists(), 'Preserve sealed predictions'
    assert read(SPEC)['shape_exponent'] == 4
    seal, cache = read_source()
    panel, work = verify_sources()
    estimates = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        tasks = ((*args, cache[args[0]['uid']]) for args in work)
        for frame, result in zip(panel, pool.map(replay_one, tasks, chunksize=3), strict=True):
            estimates[frame['uid']] = result
            if len(estimates) % 64 == 0:
                print(f'Predicted {len(estimates)} frames', flush=True)
    assert set(estimates) == set(cache) and len(estimates) == seal['frames'] == 416
    counts = Counter(r['status'] for r in estimates.values())
    path = PRIVATE / 'predictions.json'
    write(path, estimates)
    write(OUT / 'prediction_seal.json', dict(protocol=digest(SPEC), predictions=digest(path),
        implementation={name: digest(HERE / name) for name in ('predict_proxy.py', 'shape_proxy.py')},
        source_seal=digest(SOURCE_SEAL), panel=seal['panel'], frames=len(estimates),
        matched_fit_and_selection_replayed=counts['fitted'], prior_boxes_exact=len(estimates),
        stages=dict(counts), retained_withheld_raw=seal['ray_isolation']['retained_withheld_raw'],
        replayed_qwen_replies=seal['replayed_qwen_replies'], qwen_calls=0, source_training=0, target_updates=0,
        elapsed_seconds=perf_counter() - started, target_labels_used=False, FINAL_access=False))
    print(f'Sealed {len(estimates)} frames {dict(counts)}', flush=True)


if __name__ == '__main__':
    run()
