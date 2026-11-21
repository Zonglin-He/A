"""First camera-ray intersection with a fixed quartic superellipsoid."""

import numpy as np

from feasible_geometry import CameraRays


def quartic_value(points):
    squared = points * points
    return np.einsum('ij,ij->i', squared, squared)


class QuarticRays(CameraRays):
    """Use the enclosing box only to bound the intersection search."""

    def entries(self, centers):
        centers = np.atleast_2d(centers)
        entry, box_hit, near = super().entries(centers)
        hit = np.zeros_like(box_hit)
        depth = np.full_like(entry, np.inf)
        candidate, ray = np.nonzero(box_hit)
        if not len(ray):
            return depth, hit, near

        delta = centers[candidate] - self.origin
        far = np.where(self.parallel[ray], np.inf,
                       delta * self.inv_slope[ray] + self.half_depth[ray]).min(axis=1)
        start = entry[candidate, ray]
        span = np.maximum(0., far - start)
        direction = self.directions[ray]
        x0 = (start[:, None] * direction - delta) / self.half
        step = span[:, None] * direction / self.half

        # The fourth-power sum is convex along each ray. First locate its minimum.
        low, high = np.zeros_like(span), np.ones_like(span)
        for _ in range(24):
            middle = (low + high) / 2.
            point = x0 + middle[:, None] * step
            derivative = np.einsum('ij,ij->i', point * point * point, step)
            go_right = derivative < 0.
            low = np.where(go_right, middle, low)
            high = np.where(go_right, high, middle)
        minimum = (low + high) / 2.
        intersects = quartic_value(x0 + minimum[:, None] * step) <= 1. + 1.e-12

        # Find the earliest root on the decreasing segment for intersecting rays.
        candidate, ray = candidate[intersects], ray[intersects]
        x0, step = x0[intersects], step[intersects]
        high = minimum[intersects]
        low = np.zeros_like(high)
        for _ in range(24):
            middle = (low + high) / 2.
            outside = quartic_value(x0 + middle[:, None] * step) > 1.
            low = np.where(outside, middle, low)
            high = np.where(outside, high, middle)
        depth[candidate, ray] = start[intersects] + high * span[intersects]
        hit[candidate, ray] = True
        return depth, hit, near
