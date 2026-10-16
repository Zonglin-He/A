"""Frozen same-pool association diagnostics; no GT or learned parameters."""
import hashlib
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, roi_align


@torch.no_grad()
def visual_roi_features(feature_maps, normalized_cxcywh):
    """Pure backbone outputs BEFORE language/position fusion; batch size one."""
    boxes = torch.as_tensor(normalized_cxcywh, device=feature_maps[0][0].device,
                            dtype=torch.float32).reshape(-1, 4)
    xy = box_convert(boxes, 'cxcywh', 'xyxy').clamp(0, 1)
    vectors, shapes = [], []
    for fmap, mask in feature_maps:
        assert fmap.shape[0] == mask.shape[0] == 1
        height = int(mask[0].any(1).sum())
        width = int(mask[0].any(0).sum())
        expected = torch.zeros_like(mask[0]); expected[:height, :width] = True
        assert torch.equal(expected, mask[0]), 'Nonrectangular pixel mask'
        scale = xy.new_tensor([width, height, width, height])
        pooled = roi_align(fmap.float(), [xy * scale], output_size=(3, 3),
                           spatial_scale=1., sampling_ratio=2, aligned=True).mean((-1, -2))
        vectors.append(F.normalize(pooled, dim=-1, eps=1e-12))
        shapes.append(dict(shape=list(fmap.shape), valid_hw=[height, width]))
    merged = F.normalize(torch.cat(vectors, dim=-1), dim=-1, eps=1e-12)
    assert torch.isfinite(merged).all()
    return merged.cpu(), shapes


def shuffle_features(features, key, seed):
    """Preserve each frame's feature multiset, destroy proposal correspondence."""
    f = np.asarray(features, dtype=np.float64)
    if len(f) < 2:
        return f.copy(), list(range(len(f)))
    value = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest(), 16)
    order = np.roll(np.arange(len(f)), 1 + value % (len(f)-1))
    return f[order], order.tolist()


def chain_path(unaries, features, links, weight):
    """Exact global maximizer of normalized unary + adjacent cosine.

    `links[i]` connects nonempty candidate sets i and i+1; rejected observation
    gaps must be represented as False. All inputs contain only allowed boxes.
    """
    n = len(unaries)
    if len(features) != n or len(links) != max(0, n-1) or not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid chain dimensions/weight')
    if not n:
        return [], dict(objective=0., unary=0., appearance=None, edges=0)
    u = [np.asarray(a, dtype=np.float64) for a in unaries]
    f = [np.asarray(a, dtype=np.float64) for a in features]
    for a, b in zip(u, f):
        if a.ndim != 1 or not len(a) or b.ndim != 2 or len(a) != len(b):
            raise ValueError('Empty or malformed candidate set')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite chain input')
    f = [a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12) for a in f]
    count = sum(bool(x) for x in links)
    dp, back = u[0]/n, []
    similarities = []
    for i in range(1, n):
        sim = np.clip(f[i-1] @ f[i].T, -1., 1.)
        similarities.append(sim)
        edge = weight/max(count, 1) * sim if links[i-1] else np.zeros_like(sim)
        values = dp[:, None] + edge
        parent = values.argmax(0)
        dp = values[parent, np.arange(len(u[i]))] + u[i]/n
        back.append(parent)
    path = [int(dp.argmax())]
    for parent in reversed(back):
        path.append(int(parent[path[-1]]))
    path.reverse()
    unary = float(np.mean([a[j] for a, j in zip(u, path)]))
    app = float(np.mean([sim[path[i], path[i+1]] for i, sim in enumerate(similarities)
                         if links[i]])) if count else None
    objective = unary + weight*(app if app is not None else 0.)
    assert abs(objective-float(dp.max())) < 1e-10
    return path, dict(objective=objective, unary=unary, appearance=app, edges=count)
