"""Center containment intervals and fixed-pose camera-ray inference."""
from itertools import product

import numpy as np
from scipy.optimize import minimize


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def bounds(points, dims, trim=0., tolerance=0.):
    """Axiswise center interval in local coordinates, even when it is empty."""
    qlo, qhi = np.quantile(points, [trim, 1.-trim], axis=0)
    return qhi-dims/2.-tolerance, qlo+dims/2.+tolerance


def inside(center, low, high):
    return np.all((center >= low-1.e-10) & (center <= high+1.e-10), axis=-1)


class CameraRays:
    """Batch slab test in the hypothesized box basis, with optical depth as t."""
    def __init__(self, points, world_to_camera, yaw, dims, cap):
        inverse = np.linalg.inv(world_to_camera)
        axes = rotation(yaw)
        self.origin = inverse[:3, 3] @ axes
        self.directions = (points / points[:, 2:3]) @ inverse[:3, :3].T @ axes
        self.range = points[:, 2]
        self.half = np.asarray(dims) / 2.
        self.parallel = np.abs(self.directions) <= 1.e-12
        self.inv_slope = np.divide(1., self.directions, out=np.zeros_like(self.directions), where=~self.parallel)
        self.half_depth = self.half * np.abs(self.inv_slope)
        self.cap = cap

    def entries(self, centers):
        delta = np.atleast_2d(centers)[:, None, :] - self.origin
        middle = delta * self.inv_slope
        near = np.where(self.parallel, -np.inf, middle-self.half_depth)
        far = np.where(self.parallel, np.inf, middle+self.half_depth)
        entry = near.max(axis=2)
        hit = ((far.min(axis=2)+1.e-10 >= entry) & (entry > 0.)
               & ~np.any(self.parallel & (np.abs(delta) > self.half+1.e-10), axis=2))
        return entry, hit, near

    def losses(self, centers):
        entry, hit, _ = self.entries(centers)
        return np.where(hit, np.minimum(np.abs(entry-self.range), self.cap), self.cap).mean(axis=1)


def ordered(cost, points, anchor):
    return np.lexsort((np.linalg.norm(points-anchor, axis=1), np.round(cost, 12)))


def continuous_fit(rays, low, high, anchor, n_grid, max_eval):
    """Same sampled starts and local continuous optimizer for each search envelope."""
    grid = np.array(list(product(*[np.linspace(a,b,n_grid) for a,b in zip(low,high)])))
    starts = np.unique(np.vstack((grid, np.clip(anchor,low,high))), axis=0)
    samples = [starts]
    costs = [rays.losses(starts)]
    use = high-low > 1.e-9
    seeds = ordered(costs[0],starts,anchor)[:2]
    if use.any():
        for seed in seeds:
            initial = starts[seed].copy()
            tested, evaluated = [], []

            def loss(value):
                center = initial.copy()
                center[use] = value
                center = np.clip(center,low,high)
                v = float(rays.losses(center)[0])
                tested.append(center)
                evaluated.append(v)
                return v

            minimize(loss, initial[use], method='Powell', bounds=list(zip(low[use],high[use])),
                     options={'maxfev':max_eval, 'xtol':1.e-3, 'ftol':1.e-6})
            if tested:
                samples.append(np.asarray(tested))
                costs.append(np.asarray(evaluated))
    tested, c = np.concatenate(samples), np.concatenate(costs)
    best = ordered(c,tested,anchor)[0]
    return tested[best], float(c[best]), tested, c


def observe_axes(rays, mode, low, high, prior, tolerance, min_face_rays, minimum):
    """Use the prior in weak axes inside a local first-face preserving interval."""
    entry, hit, near = (x[0] for x in rays.entries(mode))
    face = np.argmax(near, axis=1)
    tied = np.sum(np.abs(near-entry[:, None]) <= 1.e-9, axis=1)>1
    inlier = hit & ~tied & (np.abs(entry-rays.range) <= tolerance)
    counts = np.bincount(face[inlier], minlength=3)
    strong = counts >= min_face_rays
    l, h = low.copy(), high.copy()
    if inlier.any():
        # Contain each fitted intersection, then bound how far in front its active face can move.
        vertices = rays.origin + entry[inlier, None]*rays.directions[inlier]
        il, ih = bounds(vertices,2.*rays.half)
        l, h = np.maximum(l, il), np.minimum(h, ih)
        for axis in range(3):
            s = inlier & (face==axis)
            d = rays.directions[s, axis]
            v = rays.origin[axis] + (entry[s]-tolerance)*d
            l[axis] = max(l[axis], (v[d>0]+rays.half[axis]).max(initial=-np.inf))
            h[axis] = min(h[axis], (v[d<0]-rays.half[axis]).min(initial=np.inf))
    assert inside(mode,l,h), (mode,l,h)
    l, h = np.minimum(l,mode), np.maximum(h,mode)
    wanted = np.where(strong, mode, np.clip(prior,l,h))
    accepted = mode.copy()
    for alpha in 2.**-np.arange(10):
        trial = mode + alpha*(wanted-mode)
        new_entry, new_hit, _ = (x[0] for x in rays.entries(trial))
        if (new_hit[inlier].all() and (np.abs(new_entry[inlier]-entry[inlier]) <= tolerance+1.e-8).all()
            and rays.losses(trial)[0] <= minimum+tolerance+1.e-10):
            accepted = trial
            break
    assert inside(accepted,low,high)
    np.testing.assert_array_equal(accepted[strong],mode[strong])
    actual = rays.losses(accepted)[0]
    return accepted, float(actual), counts, h-l, strong


def line_witnesses(rays, mode, low, high, limit):
    """Sample one-dimensional slices through a feasible near-loss mode."""
    witnesses = []
    for a in range(3):
        for end in (low[a],high[a]):
            delta = mode.copy()
            delta[a] = end
            if rays.losses(delta)[0] <= limit:
                witnesses.append(delta)
                continue
            good = mode.copy()
            for _ in range(8):
                x = (good+delta)/2.
                if rays.losses(x)[0] <= limit:
                    good = x
                else:
                    delta = x
            witnesses.append(good)
    return np.array(witnesses)
