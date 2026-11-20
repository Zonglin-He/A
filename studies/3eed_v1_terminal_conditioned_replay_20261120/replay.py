"""Run the original six 3EED decoder layers under controlled geometric schedules."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAY = HERE.parent / '3eed_v1_language_displacement_rays_20261120'
sys.path.insert(0, str(RAY))
import capture as archive
from crossplatform.candidate_frontend import CandidateFrontend
from scope183 import atomic, digest, read

WORK = HERE / 'run_PRIVATE'
PUBLIC = HERE / 'results'
CONFIG = read(HERE / 'PROTOCOL.json')
SCHEDULES = ('native', 'terminal', 'initial', 'shuffled', 'last_only')
FEATURES = ('native', 'terminal', 'initial', 'shuffled', 'last_only', 'pooled', 'geometry_text')


def frame_inputs(platform, example):
    from model51 import SENTINEL
    from runner_helpers import sample_points
    from utils.transform_waymo import transform_to_front_view
    points, _, _ = sample_points(dict(platform=platform, seed=example['seed'], pointpath=example['points']))
    if platform == 'waymo':
        points, _ = transform_to_front_view(points, np.empty((0, 7)), example['waymo_view'])
    return dict(point_clouds=torch.as_tensor(np.ascontiguousarray(points, dtype=np.float32), device='cuda')[None],
                text=[example['text'].strip()+SENTINEL])


def permutation(identity, count):
    seed = int.from_bytes(hashlib.sha256(('terminal-replay|'+identity).encode()).digest()[:8], 'big')
    order = np.random.default_rng(seed).permutation(count)
    indices = np.empty_like(order)
    indices[order] = np.roll(order, 1)
    assert (indices != np.arange(count)).all()
    return indices


def pool_tokens(points, memory, boxes):
    # PCA boxes and seed coordinates are both in the original decoder's coordinate convention.
    delta = points[None]-boxes[:, None, :3]
    cos, sin = boxes[:, 6:7].cos(), boxes[:, 6:7].sin()
    x = delta[:, :, 0]*cos + delta[:, :, 1]*sin
    y = -delta[:, :, 0]*sin + delta[:, :, 1]*cos
    local = torch.stack((x, y, delta[:, :, 2]), dim=-1)
    membership = (local.abs() <= boxes[:, None, 3:6]*.5+1e-7).all(-1)
    count = membership.sum(-1)
    return membership.float() @ memory / count[:, None].clamp_min(1), count


def feature_array(tensor):
    value = tensor.detach().cpu().numpy().copy()
    assert np.isfinite(value).all()
    return value


@torch.inference_mode()
def measure(model, inputs, slate, boxes, identity, verify_trajectory=None):
    from model51 import firstnoun_mask, native_matching
    assert len(model.decoder) == 6 and model.self_position_embedding == 'loc_learned'
    assert not any(layer.training for layer in model.modules())
    snapshots, hidden = [], []
    with ExitStack() as hooks:
        for layer in model.decoder:
            hooks.enter_context(layer.register_forward_pre_hook(
                lambda m, args, kw: snapshots.append((args, dict(kw))), with_kwargs=True))
            hooks.enter_context(layer.register_forward_hook(lambda m, args, result: hidden.append(result)))
        raw = model(inputs)
    positions = archive.trajectory(raw)
    assert positions.shape == (7, 256, 6) and len(snapshots) == len(hidden) == 6
    torch.testing.assert_close(torch.stack([args[3][0] for args, _ in snapshots]), positions[:-1], rtol=0, atol=0)
    if verify_trajectory is not None:
        np.testing.assert_array_equal(feature_array(positions), verify_trajectory)
    ids = torch.as_tensor(slate.astype('int64'), device=positions.device)
    assert len(ids.unique()) == len(ids) == 16
    reference_box = torch.as_tensor(boxes.astype('float32'), device=positions.device)
    torch.testing.assert_close(reference_box[:, :3], positions[-1, ids, :3], rtol=0, atol=0)
    # PCA may swap the horizontal lengths and floor invalid dimensions. The replay
    # always uses the untouched six native regression outputs retained above.
    raw_ids = raw['query_points_sample_inds'][0].long()
    assert raw_ids.unique().numel() == 256
    torch.testing.assert_close(raw['fp2_xyz'][0, raw_ids], raw['query_points_xyz'][0], rtol=0, atol=0)
    mask = firstnoun_mask(model, inputs['text'][0])[0].to(positions.device)
    shuffle = torch.as_tensor(permutation(identity, len(raw_ids)), device=positions.device)
    native_hidden = hidden[-1]
    features, odds, elapsed = {}, {}, {}
    parity_errors = []
    for name in SCHEDULES:
        begin = 5 if name == 'last_only' else 0
        query = snapshots[begin][0][0]
        torch.cuda.synchronize()
        tick = perf_counter()
        for k in range(begin, 6):
            args, extra = snapshots[k]
            if name == 'native':
                pos = args[3]
            elif name == 'initial':
                pos = positions[0][None]
            elif name == 'shuffled':
                pos = positions[-1, shuffle][None]
            else:
                pos = positions[-1][None]
            query = model.decoder[k](query, args[1], args[2], pos, *args[4:], **extra)
            if name == 'native':
                error = float((query-hidden[k]).abs().max())
                parity_errors.append(error)
                if error > 1e-6:
                    raise ValueError(f'Native decoder replay differs at layer {k}: {error}')
        torch.cuda.synchronize()
        elapsed[name] = perf_counter()-tick
        projected = F.normalize(model.contrastive_align_projection_image(query), p=2, dim=-1)
        if name == 'native':
            torch.testing.assert_close(projected, raw['last_proj_queries'], rtol=0, atol=1e-6)
        _, values = native_matching(projected, raw['proj_tokens'], mask)
        odds[name] = feature_array(values[ids])
        features[name] = feature_array(query[0, ids] if name == 'native' else (query-native_hidden)[0, ids])
    pooled, count = pool_tokens(raw['fp2_xyz'][0], snapshots[0][0][1][0], reference_box)
    features['pooled'] = feature_array(pooled)
    # Six geometric coordinates crossed with 48 fixed text channels. Only this
    # control sees these handcrafted feature coordinates. It has 288 coefficients.
    geom = positions[-1, ids]
    geom = torch.cat((geom[:, :3].sign()*torch.log1p(geom[:, :3].abs())/4,
                      geom[:, 3:].clamp_min(.05).log()), dim=-1)
    language = raw['text_memory'][0, mask].mean(0).reshape(48, 6).mean(-1)
    features['geometry_text'] = feature_array((geom[:, :, None]*(1+language)).flatten(1))
    assert all(v.shape == (16, 288) for v in features.values())
    return dict(features=np.stack([features[k] for k in FEATURES]),
                odds=np.stack([odds[k] for k in SCHEDULES]), timing=np.array([elapsed[k] for k in SCHEDULES]),
                parity=np.array(parity_errors), point_counts=feature_array(count),
                terminal_delta_norm=feature_array((positions[-1,ids]-positions[0,ids]).norm(dim=-1)),
                seed=feature_array(raw_ids), boxes=boxes, slate=slate)


def save_arrays(path, **values):
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, **values)
    path.with_suffix('.tmp').replace(path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k] for k in values.files}
