"""GT-free time coverage and anchor-constrained trajectory reconstruction.

The energy solver is an independent numerical implementation of absolute
linear interpolation, NOT a distinct learned method or spatial parameter TTA.
"""
import numpy as np
import torch
from methods.decota_refine_v1.api import refine, _apply
from methods.decota_s_v1.api import valid_boxes


def _grid(frame_ids, extent, k):
    ids = np.asarray(frame_ids, dtype=np.float64)
    if ids.ndim != 1 or not len(ids) or not np.isfinite(ids).all() or not (np.diff(ids) > 0).all():
        raise ValueError('Expected finite strictly increasing original frame positions')
    if len(extent) != 2 or any(isinstance(x, bool) or int(x) != x for x in extent):
        raise ValueError('Invalid extent')
    a, b = map(int, extent)
    if not 0 <= a <= b < len(ids):
        raise ValueError('Extent outside grid')
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError('Budget must be a nonnegative integer')
    return ids, a, b, min(k, b-a+1)


def coverage_radius(frame_ids, extent, positions):
    ids, a, b, _ = _grid(frame_ids, extent, len(positions))
    if not positions:
        return None  # No coverage, not a radius of zero.
    if not set(positions) <= set(range(a, b+1)):
        raise ValueError('Selected position outside extent')
    return float(np.abs(ids[a:b+1, None] - ids[list(positions)][None, :]).min(1).max())


def uniform_positions(frame_ids, extent, k=8):
    """Same physical-frame bin midpoints as the previously tested uniform rule.

The interval is [first frame_id, last frame_id + 1), in original frame units.
Empty bins are filled by the farthest currently uncovered available position.
"""
    ids, a, b, k = _grid(frame_ids, extent, k)
    if not k:
        return []
    edges = np.linspace(ids[a], ids[b]+1, k+1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = [i for i in range(a, b+1) if lo <= ids[i] < hi]
        if pool:
            out.append(min(pool, key=lambda i: (abs(ids[i]-(lo+hi)/2), i)))
    while len(out) < k:
        rest = [i for i in range(a, b+1) if i not in out]
        out.append(max(rest, key=lambda i: (min(abs(ids[i]-ids[j]) for j in out), -i)))
    return sorted(out)


def minimax_positions(frame_ids, extent, k=8):
    """Exact discrete 1-D K-center on the available frame positions.

For radius R, cover the leftmost uncovered demand using the rightmost allowed
center within R of it. This greedily reaches farthest right; feasibility uses
the fewest centers. Binary search the finite set of pairwise distances. Ties
are deterministic; leftover budget uses farthest uncovered points. Multiple
optimal sets exist: no semantic or GT criterion is used to break ties.
"""
    ids, a, b, k = _grid(frame_ids, extent, k)
    x = ids[a:b+1]
    if not k:
        return [], dict(radius=None, lower_radius=None, lower_required_centers=None)
    radii = np.unique(np.abs(x[:, None]-x[None, :]))

    def centers(radius):
        pos, out = 0, []
        while pos < len(x):
            center = int(np.searchsorted(x, x[pos]+radius, side='right')-1)
            out.append(center)
            pos = int(np.searchsorted(x, x[center]+radius, side='right'))
        return out

    lo, hi = 0, len(radii)-1
    while lo < hi:
        mid = (lo+hi)//2
        if len(centers(radii[mid])) <= k:
            hi = mid
        else:
            lo = mid+1
    out = centers(radii[lo])
    while len(out) < k:
        rest = [i for i in range(len(x)) if i not in out]
        out.append(max(rest, key=lambda i: (min(abs(x[i]-x[j]) for j in out), -i)))
    out = sorted(i+a for i in out)
    radius = coverage_radius(ids, (a, b), out)
    assert radius <= radii[lo]
    lower_count = len(centers(radii[lo-1])) if lo else None
    assert lower_count is None or lower_count > k
    return out, dict(radius=radius, lower_radius=float(radii[lo-1]) if lo else None,
                     lower_required_centers=lower_count, optimal=True, budget=k)


def select_positions(frame_ids, extent, k=8, mode='uniform'):
    if mode == 'uniform':
        return uniform_positions(frame_ids, extent, k)
    if mode == 'minimax':
        return minimax_positions(frame_ids, extent, k)[0]
    raise ValueError(mode)


def trajectory_energy(values, times):
    return float((np.diff(values, axis=0)**2 / np.diff(times)[:, None]).sum())


def solve_anchor_energy(times, positions, targets):
    """Independent FP64 banded Dirichlet solve; returns ONLY the anchor hull.

min sum ||B[i+1]-B[i]||^2 / dt[i], subject to the given anchor coordinates.
No call to np.interp or the closed-form linear interpolation implementation.
"""
    from scipy.linalg import solve_banded
    times = np.asarray(times, dtype=np.float64)
    pos = np.asarray(positions, dtype=int)
    target = np.asarray(targets, dtype=np.float64)
    if len(pos) < 2 or not (np.diff(pos)>0).all() or not (np.diff(times)>0).all():
        raise ValueError('Need at least two ordered anchors and a strictly increasing grid')
    if target.shape != (len(pos), 4) or not np.isfinite(target).all():
        raise ValueError('Invalid anchor array')
    if pos[0] < 0 or pos[-1] >= len(times):
        raise ValueError('Anchor outside grid')
    hull = np.arange(pos[0], pos[-1]+1)
    values = np.empty((len(hull), 4), dtype=np.float64)
    values[pos-pos[0]] = target
    max_stationarity = 0.
    for left, right, bl, br in zip(pos[:-1], pos[1:], target[:-1], target[1:]):
        m = right-left-1
        if not m:
            continue
        w = 1./np.diff(times[left:right+1])
        diagonal = w[:-1]+w[1:]
        bands = np.zeros((3, m), dtype=np.float64)
        bands[1] = diagonal
        bands[0, 1:] = -w[1:-1]
        bands[2, :-1] = -w[1:-1]
        rhs = np.zeros((m, 4)); rhs[0] += w[0]*bl; rhs[-1] += w[-1]*br
        inside = solve_banded((1, 1), bands, rhs)
        values[left-pos[0]+1:right-pos[0]] = inside
        full = np.vstack((bl, inside, br))
        stationarity = w[:-1, None]*(inside-full[:-2]) + w[1:, None]*(inside-full[2:])
        max_stationarity = max(max_stationarity, float(np.abs(stationarity).max()))
    return hull, values, dict(energy=trajectory_energy(values, times[hull]),
                             stationarity_max=max_stationarity,
                             anchor_error=float(np.abs(values[pos-pos[0]]-target).max()))


@torch.no_grad()
def minimum_energy_refine(base, pseudo, frame_ids):
    """Matched fallback/clipping with the frozen historical absolute baseline."""
    base = base.detach().cpu().float()
    ids, _, _, _ = _grid(frame_ids, (0, len(base)-1), 0)
    if not pseudo:
        return base.clone(), dict(mode='minimum_energy', changed_support=0, fallback='empty_supervision')
    pp = sorted(pseudo, key=lambda p: p['position'])
    pos = [p['position'] for p in pp]
    if len(set(pos)) != len(pos) or min(pos)<0 or max(pos)>=len(base):
        raise ValueError('Invalid anchor positions')
    targets = np.asarray([p['box'] for p in pp], dtype=np.float32)
    if not valid_boxes(torch.from_numpy(targets)):
        return base.clone(), dict(mode='minimum_energy', changed_support=0, fallback='invalid_pseudo_boxes')
    if len(pos)==1:
        out = base.clone();out[pos] = torch.from_numpy(targets)
        return out, dict(mode='minimum_energy', changed_support=1, single_anchor=True)
    hull, values, audit = solve_anchor_energy(ids, pos, targets)
    out, invalid = _apply(base, hull, values)
    return out, dict(mode='minimum_energy', changed_support=len(hull), invalid=invalid, **audit)


def reconstruct(base, pseudo, frame_ids, mode='absolute'):
    if mode == 'minimum_energy':
        return minimum_energy_refine(base, pseudo, frame_ids)
    return refine(base, pseudo, frame_ids, mode)
