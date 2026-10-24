"""Label-free parameterized SimplePath; no files, models or annotations."""
import types
import numpy as np
import torch
import torch.nn.functional as F
from tools.simplepath_tube.core import spatial as original
from tools.simplepath_tube.core.math import occupancy, fit_axis_affine
from stvg.predictor import output_writeback


def geometry(capture, variant, gamma):
    """Private globals: never monkeypatch the active method's functions."""
    def affinity(normalized, valid, target, source):
        ti = torch.nonzero(valid[target], as_tuple=False).flatten()
        si = torch.nonzero(valid[source], as_tuple=False).flatten()
        matrix = torch.zeros(normalized.shape[1], normalized.shape[1], dtype=torch.float64)
        if len(ti) and len(si):
            cosine = normalized[target, ti] @ normalized[source, si].T
            scores, indices = torch.topk(cosine, min(variant['topk'], len(si)), dim=1, sorted=False)
            weights = (scores / variant['temperature']).softmax(1).double()
            weights /= weights.sum(1, keepdim=True)
            matrix[ti[:, None].expand_as(indices), si[indices]] = weights
        return matrix
    fn = original.SpatialGeometry.__init__.__wrapped__
    scope = dict(fn.__globals__, affinity=affinity)
    constructor = types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)
    constructor.__kwdefaults__ = fn.__kwdefaults__
    obj = original.SpatialGeometry.__new__(original.SpatialGeometry)
    constructor(obj, capture['evidence'], output_writeback(capture['Source'], capture['raw'], gamma), capture['native_boxes'])
    return obj


def gamma_view(g, capture, gamma):
    """Reuse fixed affine candidates; recompute changed E appearance and links."""
    boxes = g.boxes.clone()
    E = torch.from_numpy(output_writeback(capture['Source'], capture['raw'], gamma)).double()
    boxes[:, 0] = E
    for side in range(2):
        boxes[~g.side_available[:, side], side + 1] = E[~g.side_available[:, side]]
    occ, _, _ = occupancy(boxes, g.height, g.width)
    occ *= capture['evidence'].valid.reshape(g.frames, 1, g.height, g.width)
    raw = capture['evidence'].features.double()
    pooled = torch.bmm(occ.flatten(-2), raw) / occ.flatten(-2).sum(-1).clamp_min(1e-15)[..., None]
    features = F.normalize(pooled, dim=-1, eps=1e-12)
    association = torch.einsum('tkd,tsd->tks', features, g.side_prototypes).clamp(0, 1)
    w = g.side_available.double()
    u = ((association * w[:, None]).sum(-1) / w.sum(-1).clamp_min(1e-15)[:, None]).numpy()
    b = boxes.numpy()
    dz = np.concatenate(((b[..., :2]-E.numpy()[:, None, :2])/E.numpy()[:, None, 2:],
                         np.log(b[..., 2:]/E.numpy()[:, None, 2:])), -1)
    guard = -np.expm1(-(np.linalg.norm(dz, axis=-1)/2)**2)
    return b, u, original.box_links(boxes).numpy(), guard


def batch_dp(boxes, unary, pair, valid):
    """Exact DP for many profiles, native first-index tie order."""
    P, T, K = unary.shape
    values = np.where(valid[:, 0], unary[:, 0], -np.inf)
    back = np.empty((P, T-1, K), np.int8)
    for t in range(1, T):
        scores = values[:, :, None] + pair[:, t-1] + unary[:, t, None, :]
        back[:, t-1] = scores.argmax(1)
        values = np.where(valid[:, t], scores.max(1), -np.inf)
    actions = np.empty((P, T), np.int8)
    actions[:, -1] = values.argmax(1)
    for t in range(T-2, -1, -1):
        actions[:, t] = back[np.arange(P), t, actions[:, t+1]]
    return boxes[np.arange(T)[None], actions], actions


def finish(source, predicted, observed, params):
    alpha = np.where(observed, params['alpha_observed'], params['alpha_unobserved'])[:, None]
    z = np.concatenate((source[:, :2], np.log(source[:, 2:])), -1)
    target = np.concatenate((predicted[:, :2], np.log(predicted[:, 2:])), -1)
    mixed = z + alpha * (target-z)
    out = np.concatenate((mixed[:, :2], np.exp(mixed[:, 2:])), -1)
    out = np.where(alpha == 0, source, np.where(alpha == 1, predicted, out))
    out[:, 2] *= params['width_scale']
    out[:, 3] *= params['height_scale']
    assert np.isfinite(out).all() and (out[:, 2:] > 0).all()
    return out


def predictions(capture, fixed, profiles, variants):
    source = np.asarray(capture['Source'], np.float64)
    ids = np.asarray(capture['frame_ids']); T = len(ids)
    observed = np.zeros(T, bool); observed[capture['positions8']] = True
    interval = np.asarray(fixed['Selected_interval'])
    event = (ids >= interval[0]) & (ids < interval[1])
    out = np.empty((len(profiles), T, 4), np.float64)
    for i, p in enumerate(profiles):
        if p['family'] == 'source_control':
            out[i] = source
        elif p['family'] == 'source_calibration':
            out[i] = finish(source, source, observed, p)
        elif p['family'] == 'sealed_full_blend':
            out[i] = finish(source, np.asarray(fixed['Full'], np.float64), observed, p)
        elif p['family'] == 'initialization':
            E = output_writeback(source, capture['raw'], p['gamma'])
            out[i] = finish(source, E, observed, p)
    geometries = sorted({p['geometry'] for p in profiles if p['family'] == 'simplepath'})
    baseline_checked = False
    for v in geometries:
        g = geometry(capture, variants[v], 0.)
        for gamma in sorted({p['gamma'] for p in profiles if p['family'] == 'simplepath' and p['geometry'] == v}):
            indexes = [i for i,p in enumerate(profiles) if p['family']=='simplepath' and p['geometry']==v and p['gamma']==gamma]
            b, u, link, penalty = gamma_view(g, capture, gamma)
            pp = [profiles[i] for i in indexes]; P = len(pp)
            scalar = lambda k: np.array([p[k] for p in pp])
            supported = g.supported.numpy()
            obs = event & observed; unobs = event & ~observed
            weights = (scalar('observed_weight')[:, None] * obs/max(1, obs.sum())
                       + scalar('unobserved_weight')[:, None] * (unobs & supported)/max(1, unobs.sum()))
            unary = weights[..., None] * u[None] - scalar('guard')[:, None, None] * penalty[None]/T
            edges = event[:-1] & event[1:] & supported[:-1] & supported[1:]
            edge_w = scalar('link_weight')[:, None] * edges[None]/max(1, (event[:-1]&event[1:]).sum())
            pair = edge_w[:, :, None, None] * link[None]
            valid = np.broadcast_to(g.valid.numpy(), (P,T,3)).copy()
            active = event & supported
            valid[:, ~active, 1:] = False
            margin = u[:, 1:] - u[:, :1]
            valid[:, :, 1:] &= margin[None] >= scalar('admission_margin')[:, None, None]
            pred, actions = batch_dp(b, unary, pair, valid)
            for j, i in enumerate(indexes):
                out[i] = finish(source, pred[j], observed, profiles[i])
                if profiles[i].get('original_baseline'):
                    assert np.array_equal(pred[j], fixed['Full']), 'Original SimplePath replay changed'
                    assert np.array_equal(out[i], fixed['Full']), 'Baseline output changed'
                    baseline_checked = True
    assert baseline_checked and np.array_equal(out[0], source)
    return out


def dense_scores(predictions, ids, width, height, truth, clip=False):
    """Vectorized same dense metric; annotations used only by offline scorer."""
    fids = np.array(sorted(map(int, truth)))
    gt = np.array([truth[str(f)] if str(f) in truth else truth[int(f)] for f in fids])
    boxes = np.concatenate((predictions[..., :2]-predictions[..., 2:]/2,
                            predictions[..., :2]+predictions[..., 2:]/2), -1)
    boxes *= np.array([width,height,width,height])
    if clip: boxes = np.maximum(boxes, 0)
    left = np.clip(np.searchsorted(ids, fids, side='right')-1, 0, len(ids)-2)
    a = np.clip((fids-ids[left])/(ids[left+1]-ids[left]),0,1)
    b = boxes[:, left]*(1-a)[None,:,None]+boxes[:,left+1]*a[None,:,None]
    inter = np.maximum(np.minimum(b[...,2:],gt[None,:,2:])-np.maximum(b[...,:2],gt[None,:,:2]),0).prod(-1)
    union = np.maximum(b[...,2:]-b[...,:2],0).prod(-1)+np.maximum(gt[:,2:]-gt[:,:2],0).prod(-1)[None]-inter
    iou = np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    iou[:,(fids<ids[0])|(fids>ids[-1])] = 0
    return fids, iou
