"""Fixed donor intervention and frozen linear readout; no training functions."""
import hashlib
import numpy as np
from scripts.tastvg_information_atlas_math_v1 import VIEWS, predict


def caption_norm(s):
    return ' '.join(s.casefold().split())


def donor_map(rows, dataset):
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(
        f'query-swap-v1|{dataset}|{rows[i]["source"]}'.encode()).hexdigest())
    for offset in range(1, len(order)):
        mapping = {i: order[(j + offset) % len(order)] for j, i in enumerate(order)}
        if all(rows[i]['source'] != rows[d]['source'] and
               rows[i]['input']['video_sha256'] != rows[d]['input']['video_sha256'] and
               caption_norm(rows[i]['input']['caption']) != caption_norm(rows[d]['input']['caption'])
               for i, d in mapping.items()):
            return mapping, offset
    raise ValueError('No valid deterministic cyclic caption derangement; do not silently change donor rule')


def focused_models(models):
    ans = {}
    for name, m in models.items():
        family, view, task, control = name.split('/')
        if (family == 'frame' and task == 'event') or (
                family == 'candidate' and task in ['precision', 'recall', 'tiou']):
            ans[name] = m
    assert len(ans) == 40
    return ans


def readout(feature, models):
    ids = np.asarray(feature['frame_ids'], float)
    h = feature['hidden'].double().numpy()
    position = ((ids - ids[0]) / (ids[-1] + 1 - ids[0]))[:, None]
    out = {}
    for name, m in focused_models(models).items():
        family, view, _, _ = name.split('/')
        x = (h if view == 'Hidden' else position) if family == 'frame' else (
            feature['geometry'] if view == 'Geometry' else feature['x'][:, VIEWS[view]])
        out[name] = predict(m, x)
        if family == 'frame':
            out[name + '/logit'] = (x - m['mean']) / m['std'] @ m['weight'] + m['bias']
    return out
