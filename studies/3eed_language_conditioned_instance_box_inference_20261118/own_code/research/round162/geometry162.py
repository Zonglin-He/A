"""Robust gravity-aligned continuous fitting on the observed LiDAR surface."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def axial(theta):
    return (theta + np.pi / 2) % np.pi - np.pi / 2


def axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def local_points(points, box):
    return (points - box[:3]) @ axes(box[6])


def in_box(points, box):
    return (np.abs(local_points(points, box)) <= box[3:6] / 2 + 1e-7).all(1)


def to_parameters(box):
    return np.r_[box[:3], np.log(box[3:6]), box[6]]


def to_box(params):
    return np.r_[params[:3], np.exp(params[3:6]), axial(params[6])]


def surface_cost(points, params, scale, derivative=False):
    """Pseudo-Huber loss of the signed distance to an upright box surface."""
    rotation = axes(params[6])
    local = (points - params[:3]) @ rotation
    half = np.exp(params[3:6]) / 2
    delta = np.abs(local) - half
    positive = np.maximum(delta, 0)
    outside = np.linalg.norm(positive, axis=1)
    signed = outside + np.minimum(delta.max(1), 0)
    d = signed / scale
    root = np.sqrt(1 + (d / .2) ** 2)
    loss = .2 ** 2 * (root - 1)
    if not derivative:
        return loss
    direction = np.divide(positive, outside[:, None], out=np.zeros_like(positive), where=outside[:, None] > 0)
    inside = outside == 0
    direction[inside, delta[inside].argmax(1)] = 1
    directed = direction * (d / (root * scale))[:, None]
    gx = directed * np.sign(local)
    center = -gx @ rotation.T
    size = -directed * half
    yaw = gx[:, 0] * local[:, 1] - gx[:, 1] * local[:, 0]
    return loss, np.c_[center, size, yaw]


def effective_count(weights):
    return weights.sum() ** 2 / max(float(weights @ weights), 1e-12)


def pca_quantile_box(points, initial):
    """Ordinary robust OBB around the same fixed old winner."""
    if len(points) < 8:
        return initial.copy()
    local = local_points(points, initial)
    lo, hi = np.quantile(local, [.05, .95], axis=0)
    dims = np.clip(hi - lo, initial[3:6] * .55, initial[3:6] * 1.6)
    center = initial[:3] + ((lo + hi) / 2) @ axes(initial[6]).T
    return np.r_[center, dims, initial[6]]


@dataclass
class BoxFitter:
    initial: np.ndarray
    prior_logdim: np.ndarray
    prior_sigma: np.ndarray
    size_prior: float = .012
    fixed_dims: bool = False
    missing_mass: float = 4.
    missing_cost: float = .10
    fit_iterations: int = 5

    @property
    def scale(self):
        return np.clip(np.min(self.initial[3:6]), .3, 4.)

    def regularize(self, x):
        origin = to_parameters(self.initial)
        # Wide trust limits and a weak prior stabilize unobserved faces.
        scale = np.maximum(self.initial[3:6], .35)
        step = x - origin
        drift = step[:3] / scale
        relative_dim = step[3:6]
        absolute_dim = (x[3:6] - self.prior_logdim) / self.prior_sigma
        center = .022 * (drift @ drift)
        size = self.size_prior * (.9 * (relative_dim @ relative_dim) + .1 * (absolute_dim @ absolute_dim))
        yaw = .006 * (1 - np.cos(2 * step[6]))
        grad = np.r_[.044 * drift / scale, self.size_prior * (1.8 * relative_dim + .2 * absolute_dim / self.prior_sigma),
                     .012 * np.sin(2 * step[6])]
        return float(center + size + yaw), grad

    def mean_cost(self, cost, weights):
        return float((weights @ cost + self.missing_mass * self.missing_cost) / (weights.sum() + self.missing_mass))

    def fit(self, points, weights, start, relation=None):
        if len(points) < 8 or effective_count(weights) < 6 or weights.sum() < .75:
            return start.copy(), 0
        x0 = to_parameters(self.initial)
        x = to_parameters(start)
        # Align the axial representation to the initial branch before optimization.
        x[6] = x0[6] + axial(x[6] - x0[6])
        halfspan = np.r_[.45 * np.maximum(self.initial[3:6], .30), np.full(3, 0 if self.fixed_dims else .48), 1.1]
        bounds = list(zip(x0 - halfspan, x0 + halfspan))
        den = weights.sum() + self.missing_mass

        def objective(params):
            per_point, gradients = surface_cost(points, params, self.scale, derivative=True)
            penalty, grad = self.regularize(params)
            value = self.mean_cost(per_point, weights) + penalty
            grad += weights @ gradients / den
            if relation is not None:
                value_r, grad_r = relation(params[:3])
                value += value_r
                grad[:3] += grad_r
            return value, grad

        start_value, _ = objective(x)
        result = minimize(objective, x, method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': self.fit_iterations, 'maxls': 8, 'ftol': 1e-10})
        if not np.isfinite(result.fun) or result.fun > start_value + 1e-10:
            return start.copy(), int(result.nfev)
        box = to_box(result.x)
        if self.fixed_dims:
            box[3:6] = self.initial[3:6]
        return box, int(result.nfev)


@dataclass
class CameraX:
    intrinsic: np.ndarray
    transform: np.ndarray
    distortion: np.ndarray
    width: int

    def project(self, center):
        """Normalized horizontal pixel and its derivative in model coordinates."""
        rotation = self.transform[:3, :3]
        p = rotation @ center + self.transform[:3, 3]
        if p[2] <= .1:
            return np.nan, np.zeros(3)
        if np.any(self.distortion):
            import cv2
            pixel, jac = cv2.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3),
                                           self.intrinsic, self.distortion)
            return float(pixel[0, 0, 0] / self.width), jac[0, 3:6] @ rotation / self.width
        norm = self.intrinsic[0] @ p
        grad = self.intrinsic[0] / p[2]
        grad[2] -= norm / p[2] ** 2
        return float(norm / (p[2] * self.width)), grad @ rotation / self.width


def wrong_side(difference, dead_zone):
    """Penalize a violation, saturate outliers, and leave satisfied relations flat."""
    error = max(0., difference - dead_zone) / .075
    if error == 0.:
        return 0., 0.
    den = 1 + error * error
    return error ** 2 / den, (2 * error / den ** 2) / .075
