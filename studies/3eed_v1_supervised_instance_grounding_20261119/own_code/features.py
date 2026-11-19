"""Frozen single-frame features shared by source fitting and target prediction."""

import argparse
from collections import Counter
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from scope import HERE, ROOT, PLATFORMS, PINS, file_sha, protocol, read, save, source_rows, stable
from roles import inventory, match_natural, parse_roles, project_inventory, synthetic_pair

sys.path[:0] = [str(ROOT), str(HERE.parent / '3eed_v1_retrieval_binding_refinement_20261119')]
from mechanisms import CandidateGeometry
from crossplatform.ground import Grounder
from crossplatform.candidate_frontend import WAYMO_VIEWS

FEATURES = HERE / 'features_PRIVATE'
SHARED_GEOMETRY = HERE.parent / '3eed_v1_retrieval_binding_refinement_20261119/mechanisms.py'
MAX_INVENTORY = 32


def observation(row):
    view = WAYMO_VIEWS[int(row['suffix'] or 0)] if row['platform'] == 'waymo' else 'F'
    return dict(target=row['platform'], points=row['pointpath'], rgb=row['imagepath'],
                calibration=row['metapath'], text=row['caption'], seed=protocol()['point_seed'],
                waymo_view=view)


def camera_inputs(frame, view_index):
    from crossplatform.round7.camera_projection import CameraSpec, model_to_camera_matrix, read_camera_allowlist

    fields = read_camera_allowlist(frame['calibration'])
    spec = CameraSpec.from_camera_fields(frame['target'], fields, waymo_view_suffix=view_index,
                                        missing_distortion_policy='assume_zero_declared')
    intrinsics, _, distortion, _ = spec.validated()
    with Image.open(frame['rgb']) as image:
        w, h = image.size
    # A homogeneous pinhole projection is used as a continuous feature. If
    # distortion is supplied, suppress explicit region/direction priors.
    pinhole = intrinsics @ model_to_camera_matrix(spec)[:3]
    pinhole[:2] /= np.asarray([w, h])[:, None]
    return pinhole.astype(np.float32), not bool(np.any(distortion))


@torch.inference_mode()
def encode(model, frame):
    from grounding158 import select_grounding_box
    from reader158 import location_intervals

    observed = {}
    scored, evidence = model._read_observation(frame, capture=observed)
    fixed = model._decoder(evidence)['boxes'][0].cpu().numpy()
    v1 = fixed[0]
    role = parse_roles(frame['text'].strip(), observed['graph'])
    native = observed['native']
    tokens = role.mask(observed['offsets'], observed['valid'])
    caption = frame['text'].strip()
    content = (observed['offsets'][:, 1] > observed['offsets'][:, 0])
    content &= (observed['offsets'][:, 0] < len(caption)) & observed['valid']
    masks = torch.as_tensor(np.concatenate((tokens, content[None])), device=native['xyz'].device).float()
    masks = masks / masks.sum(-1, keepdim=True).clamp_min(1.)
    text = masks @ native['text'][0].float()
    token_proj = masks @ native['tokens'][0]

    geometry = CandidateGeometry(scored.boxes, scored.sampled_points)
    anchors = np.stack([geometry.orient(j) for j in range(len(scored.boxes))])
    original = np.flatnonzero((anchors == v1).all(-1))
    if not len(original):
        raise RuntimeError('The v1 selected box is absent from the frozen proposal pool')
    npoints = protocol()['proposal']['raw_points_for_local_support']
    cloud = scored.sampled_points[:npoints]
    xyz = native['xyz'][0].cpu().numpy()
    nearest = cKDTree(xyz).query(cloud, k=1)[1]
    _, utility = select_grounding_box(scored.boxes, scored.probability)
    logprior = np.log(np.maximum(utility, 1e-100))
    logprior = (logprior - logprior.max()).clip(-12, 0)

    camera, camera_known = camera_inputs(frame, scored.view_index)
    bands = np.array([[0.,1.],[0.,1.]],np.float32)
    if camera_known:
        for axis, low, high in location_intervals(caption, role.subject):
            bands[axis] = (low, high)
    ref_phrase = caption[slice(*role.reference)].lower()
    order = int('rightmost' in ref_phrase) - int('leftmost' in ref_phrase)
    return ({
        'points': cloud.astype(np.float32), 'nearest': nearest.astype(np.int16),
        'seed_xyz': xyz.astype(np.float32), 'seed_feat': native['points'][0].cpu().numpy(),
        'seed_obj': native['objectness'][0].half().cpu().numpy(),
        'anchors': anchors, 'anchor_query': native['queries'][0].half().cpu().numpy(),
        'prior': logprior.astype(np.float32), 'text': text.half().cpu().numpy(),
        'token_proj': token_proj.half().cpu().numpy(), 'camera': camera, 'bands': bands,
        'relation': np.int8(role.relation if tokens[1].any() and camera_known else 0),
        'reference_ord': np.int8(order if camera_known else 0),
        'reference_phrase': np.bool_(tokens[1].any()), 'v1_index': np.int16(original[0]),
        'v1_box': v1.astype(np.float32), 'v1_reference_box': fixed[1].astype(np.float32),
        'v1_has_reference': np.bool_(bool(evidence['has_reference'][0])),
        'view_index': np.int8(scored.view_index),
    }, role)


def annotate(feature, boxes, subject, reference, origin):
    ordered = [subject] + ([reference] if reference >= 0 else [])
    ordered.extend(j for j in range(len(boxes)) if j not in ordered)
    known = min(len(ordered), MAX_INVENTORY)
    padded = np.zeros((MAX_INVENTORY, 7), dtype=np.float32)
    padded[:, 3:6] = 1
    padded[:known] = boxes[ordered[:known]]
    return dict(feature, gt=padded, gt_valid=np.arange(MAX_INVENTORY) < known,
                gt_reference=np.int8(1 if reference >= 0 else -1), origin=np.int8(origin))


def feature_dependencies():
    paths = [HERE / name for name in ('protocol.json', 'scope.py', 'features.py', 'roles.py')]
    paths.append(SHARED_GEOMETRY)
    paths.extend(ROOT / path for path in PINS)
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def write_block(path, records):
    cols = {name: np.stack([item[name] for item in records]) for name in records[0]}
    draft = path.with_suffix('.tmp')
    with draft.open('wb') as f:
        np.savez(f, **cols)
    os.replace(draft, path)


def extract(smoke):
    config = protocol()
    target = HERE / ('engineering_features_PRIVATE' if smoke else FEATURES.name)
    target.mkdir(mode=0o700)
    lock = dict(dependencies=feature_dependencies(),
                input_seal_sha256=file_sha(HERE/'inputs_PRIVATE/SEAL.json'),
                source_only=True, FINAL_access=False)
    save(target/'LOCK.json',lock)
    all_counts = {}
    started = perf_counter()
    for source in PLATFORMS:
        model = Grounder(source)
        for subset in ('fit', 'calibration'):
            data = source_rows(source, subset)
            cap = config['source'][f'synthetic_{subset}_cap']
            if smoke:
                # Test both generic and prospective natural-relation language.
                data = data[:3] + data[-3:]
                cap = 3
            folder = target / f'{source}_{subset}'
            folder.mkdir()
            buffered, files, counts = [], {}, Counter()
            synthetic = 0
            for index, row in enumerate(data):
                if file_sha(row['pointpath']) != row['point_sha256']:
                    raise RuntimeError('Source point file changed after registration')
                boxes, classes = inventory(row)
                feature, role = encode(model, observation(row))
                projection = project_inventory(row, boxes) if role.relation or synthetic < cap else None
                reference = match_natural(role, classes, boxes, projection) if feature['relation'] else -1
                origin = int(reference >= 0)
                buffered.append(annotate(feature, boxes, 0, reference, origin))
                counts['natural'] += 1
                counts['natural_reference_inventory_unique'] += origin
                counts['natural_reference_unresolved'] += int(reference < 0)
                if synthetic < cap and len(boxes) >= 3:
                    option = synthetic_pair(classes, projection, int(stable('synthetic', row['point_sha256'])[:12],16))
                    if option is not None:
                        caption, subject, ref, sign = option
                        extra, _ = encode(model, observation(dict(row, caption=caption)))
                        if extra['relation'] == sign:
                            buffered.append(annotate(extra, boxes, subject, ref, 2))
                            synthetic += 1
                if len(buffered) >= 24 or index + 1 == len(data):
                    path = folder / f'{len(files):04d}.npz'
                    write_block(path, buffered)
                    files[path.name] = dict(n=len(buffered), sha256=file_sha(path))
                    buffered.clear()
                if (index+1) % 80 == 0 or index+1 == len(data):
                    print(source, subset, index+1, '/',len(data),'synthetic',synthetic,
                          'unique natural references',counts['natural_reference_inventory_unique'],
                          'seconds',round(perf_counter()-started,1),flush=True)
            counts['synthetic'] = synthetic
            data_n = sum(f['n'] for f in files.values())
            assert data_n == len(data) + synthetic
            save(folder/'SEAL.json', dict(n=data_n, counts=counts, files=files,
                 input_sha256=file_sha(HERE/'inputs_PRIVATE'/f'{source}_{subset}.json'),
                 lock_sha256=file_sha(target/'LOCK.json'), FINAL_access=False))
            all_counts[folder.name] = dict(n=data_n, counts=counts, seal_sha256=file_sha(folder/'SEAL.json'))
        del model
        torch.cuda.empty_cache()
    assert feature_dependencies() == lock['dependencies']
    save(target/'SEALED.json', dict(
        splits=all_counts, elapsed_seconds=perf_counter()-started, lock_sha256=file_sha(target/'LOCK.json'),
        source_only=True, FINAL_access=False))


if __name__ == '__main__':
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument('--smoke', action='store_true')
    extract(options.parse_args().smoke)
