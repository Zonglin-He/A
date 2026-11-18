"""Alternating single-frame support assignment and continuous box inference."""
from dataclasses import dataclass, replace
import itertools
import time

import numpy as np

from geometry162 import BoxFitter, CameraX, effective_count, in_box, pca_quantile_box, surface_cost, to_parameters, wrong_side


@dataclass(frozen=True)
class Options:
    temperature: float = .85
    role_bias: float = .65
    fit_weight: float = .23
    size_prior: float = .012
    relation_weight: float = .16
    init: str = 'scene'
    graph: str = 'correct'
    path: str = 'joint'
    fixed_dims: bool = False
    missing_negative: bool = False


CALIBRATION = {
    'default': Options(),
    'wider_role_support': replace(Options(), temperature=1.10, role_bias=.35),
    'weaker_geometry_feedback': replace(Options(), fit_weight=.115),
}
CONTROLS = {
    'lcib': {},
    'serial': {'path': 'serial'},
    'single_fit': {'path': 'one'},
    'no_feedback': {'path': 'no_feedback'},
    'no_relation': {'relation_weight': 0.},
    'no_rgb_graph': {'graph': 'no_rgb_graph'},
    'swap_cross': {'graph': 'swap_cross'},
    'uniform_cross': {'graph': 'uniform_cross'},
    'bounded_origins': {'init': 'inside'},
    'expanded_origins': {'init': 'old'},
    'fixed_dimensions': {'fixed_dims': True},
    'visible_surface_only': {'size_prior': 0.},
    'missing_as_negative': {'missing_negative': True},
}
SIMPLE = ('pca_obb', 'uniform_fit', 'uniform_fixed')
MAIN = ('lcib', 'serial', 'uniform_fit', 'uniform_fixed', 'pca_obb', 'fixed_winner_support')


def project_simplex(x, allowed):
    x = np.where(allowed, x, -1e8)
    values = np.sort(x, axis=-1)[:, ::-1]
    counts = np.arange(1, x.shape[1] + 1)
    levels = (values.cumsum(axis=-1) - 1) / counts
    pivot = (values > levels).sum(1) - 1
    return np.maximum(0., x - levels[np.arange(len(x)), pivot, None])


@dataclass
class Region:
    query: int
    box: np.ndarray
    log_prior: float
    fitter: BoxFitter
    points: np.ndarray
    interpolation_index: np.ndarray
    interpolation_weight: np.ndarray
    fit_weight: float
    visible_confidence: float

    def point_weights(self, support):
        return (self.interpolation_weight * support[self.interpolation_index]).sum(1)


@dataclass
class Role:
    hypotheses: list
    prior: np.ndarray
    allowed: np.ndarray
    state: np.ndarray
    graph: object
    degree: np.ndarray


@dataclass
class Result:
    box: np.ndarray
    hypotheses: np.ndarray
    origins: np.ndarray
    role_support: np.ndarray
    initial_support: np.ndarray
    role_boxes: np.ndarray
    role_queries: np.ndarray
    result_origin: int
    old_support: np.ndarray
    fit_calls: int
    sweeps: int
    fallback: bool
    trace: list
    trace_boxes: np.ndarray
    elapsed: float


def build_roles(scene, options, log_dims, sigma):
    graph, degree, _, _ = scene.graph(options.graph)
    roles = []
    for index, mask in enumerate(scene.ev.role_mask):
        origins = scene.candidates(index, log_dims, options.init) if mask.any() else []
        if not origins:
            roles.append(None)
            continue
        prior, allowed = scene.evidence(index, origins, options)
        regions = []
        for j, hypothesis in enumerate(origins):
            box = hypothesis.box
            locked = of_query_box(scene, hypothesis.query) if options.init == 'inside' else None
            selected = scene.select_near(box, locked=locked)
            fitter = BoxFitter(box.copy(), log_dims, sigma, size_prior=options.size_prior, fixed_dims=options.fixed_dims)
            mass = prior[:scene.nseed, j]
            visible = mass[scene.ev.visible]
            coverage = float(visible.sum() / max(mass.sum(), 1e-12))
            count = effective_count(visible)
            confidence = min(1., coverage) * count / (count + 6)
            regions.append(Region(hypothesis.query, box.copy(), hypothesis.log_prior, fitter,
                                   scene.points[selected], scene.neighbor[selected], scene.interpolation[selected],
                                   options.fit_weight * float(np.clip(mass.sum(), 4., 48.)),
                                   confidence))
        roles.append(Role(regions, prior, allowed, prior.copy(), graph, degree))
    return roles


def of_query_box(scene, query):
    box = np.r_[scene.ev.boxes[query], 0.].astype(float)
    box[3:6] = np.maximum(box[3:6], 1e-6)
    return box


def preparation(role, boxes):
    return [surface_cost(h.points, to_parameters(box), h.fitter.scale)
            for h, box in zip(role.hypotheses, boxes)]


def support_energy(role, state, costs, derivative=False):
    residual = state - role.prior
    laplace = role.degree[:, None] * state - role.graph @ state
    value = .5 * float(np.sum(residual ** 2) + np.sum(state * laplace))
    gradient = residual + laplace if derivative else None
    for j, (h, cost) in enumerate(zip(role.hypotheses, costs)):
        weights = h.point_weights(state[:, j])
        fit = h.fitter.mean_cost(cost, weights)
        value += h.fit_weight * fit
        if derivative:
            grad = h.fit_weight * (cost - fit) / (weights.sum() + h.fitter.missing_mass)
            np.add.at(gradient[:, j], h.interpolation_index.ravel(),
                      (h.interpolation_weight * grad[:, None]).ravel())
    return value, gradient


def improve_support(role, boxes, count):
    costs = preparation(role, boxes)
    steps = 0
    anchor = (1. + 2. * role.degree)[:, None]
    for _ in range(count):
        val, grad = support_energy(role, role.state, costs, derivative=True)
        for ratio in (1., .5, .25):
            trial = project_simplex(role.state - ratio * grad / anchor, role.allowed)
            energy, _ = support_energy(role, trial, costs)
            if energy <= val + 1e-10:
                role.state = trial
                break
        steps += 1
    return steps


class Relationships:
    """Only verified camera-horizontal parser edges couple spatially distinct roles."""
    def __init__(self, ev, roles, weight):
        self.camera = CameraX(ev.camera_intrinsic, ev.model_to_camera, ev.distortion, ev.width)
        self.weight = weight
        self.edges = [(int(a), int(b), int(sign)) for a, b, sign in ev.edges
                      if a != b and roles[a] is not None and roles[b] is not None and weight]
        self.roles = roles
        self.priors = []
        for role in roles:
            p = np.exp([h.log_prior for h in role.hypotheses]) if role else np.empty(0)
            self.priors.append(p / (.50 + p.sum()))
        # Eligibility is measured at the unmodified starting boxes so moving
        # out of view cannot deactivate a relation cost during optimization.
        self.pairs = []
        for a, b, sign in self.edges:
            for i, first in enumerate(roles[a].hypotheses):
                x, _ = self.camera.project(first.box[:3])
                for j, second in enumerate(roles[b].hypotheses):
                    y, _ = self.camera.project(second.box[:3])
                    w = self.weight * self.priors[a][i] * self.priors[b][j]
                    w *= np.sqrt(first.visible_confidence * second.visible_confidence)
                    if w and 0 <= x <= 1 and 0 <= y <= 1:
                        self.pairs.append((a, i, b, j, sign, w))

    def neighbors(self, role_id, option, center):
        pairs = []
        for a, i, b, j, sign, weight in self.pairs:
            if (role_id, option) == (a, i):
                opponent, attempt, side = b, j, sign
            elif (role_id, option) == (b, j):
                opponent, attempt, side = a, i, -sign
            else:
                continue
            other = self.roles[opponent].hypotheses[attempt]
            x, _ = self.camera.project(other.box[:3])
            pairs.append((side, x, weight))
        if not pairs or not np.isfinite(self.camera.project(center)[0]):
            return None

        def evaluate(new_center):
            x, jac = self.camera.project(new_center)
            if not np.isfinite(x):
                return sum(w for _, _, w in pairs), np.zeros(3)
            out, derivative = 0., 0.
            for side, p, w in pairs:
                value, grad = wrong_side(side * (x - p), .015) if np.isfinite(p) else (1., 0.)
                out += w * value
                derivative += side * w * grad
            return float(out), derivative * jac
        return evaluate

    def energy(self):
        loss = 0.
        for a, i, b, j, sign, weight in self.pairs:
            x, _ = self.camera.project(self.roles[a].hypotheses[i].box[:3])
            y, _ = self.camera.project(self.roles[b].hypotheses[j].box[:3])
            value = wrong_side(sign * (x - y), .015)[0] if np.isfinite(x + y) else 1.
            loss += weight * value
        return float(loss)

    def decision_costs(self, selected, nseed):
        loss = 0.
        for a, b, sign in self.edges:
            i, j = selected[a], selected[b]
            if i < 0 or j < 0:
                continue
            ra, rb = self.roles[a], self.roles[b]
            first, second = ra.hypotheses[i], rb.hypotheses[j]
            left, right = ra.state[:nseed, i], rb.state[:nseed, j]
            same = np.minimum(left, right).sum() / max(np.maximum(left, right).sum(), 1e-12)
            # Only roles in an explicitly distinct, executable relation compete.
            loss += self.weight * max(0., (same - .35) / .65)
            x, _ = self.camera.project(first.box[:3])
            y, _ = self.camera.project(second.box[:3])
            if 0 <= x <= 1 and 0 <= y <= 1:
                value, _ = wrong_side(sign * (x - y), .015)
                conf = np.sqrt(first.visible_confidence * second.visible_confidence)
                loss += 2. * self.weight * conf * value
        return loss


def fit_roles(roles, relations):
    calls = 0
    for e, role in enumerate(roles):
        if role is None:
            continue
        for j, h in enumerate(role.hypotheses):
            weights = h.point_weights(role.state[:, j])
            term = relations.neighbors(e, j, h.box[:3])
            def scaled(center, term=term, w=h.fit_weight):
                cost, grad = term(center)
                return cost / w, grad / w
            h.box, spent = h.fitter.fit(h.points, weights, h.box, scaled if term else None)
            calls += spent
    return calls


def choose(scene, roles, relations):
    """Compare complete role assignments while retaining UNKNOWN for each role."""
    scores = []
    N = scene.nseed
    for e, role in enumerate(roles):
        goods = [(-1, -1.45 if e == 0 else -1.15)]
        if role:
            for j, h in enumerate(role.hypotheses):
                weights = h.point_weights(role.state[:, j])
                if effective_count(weights) < 6 or weights.sum() < .75:
                    continue
                support = role.state[:N, j]
                affinity = (support @ scene.ev.affinity[e]) * N / max(support.sum(), 1e-9)
                mean = h.fitter.mean_cost(surface_cost(h.points, to_parameters(h.box), h.fitter.scale), weights)
                score = .85 * h.log_prior + .43 * np.log(np.clip(affinity, .2, 8.)) - 2.5 * mean
                goods.append((j, float(score)))
        scores.append(goods)
    if not scores:
        return ()
    # Factor evaluation is applied only to endpoints with known relations; its
    # bounded penalty cannot change the validity of an independent object.
    involved = {i for a, b, _ in relations.edges for i in (a, b)}
    alternatives = [sorted(items, key=lambda p: -p[1])[:4 if r in involved else 1] for r, items in enumerate(scores)]
    beam = [((), 0.)]
    for items in alternatives:
        beam = [(old + (i,), value + score) for (old, value), (i, score) in itertools.product(beam, items)]
    best, _ = max(beam, key=lambda p: p[1] - relations.decision_costs(p[0], N))
    return best


def solve(scene, options, prior_logdim, prior_sigma):
    tic = time.perf_counter()
    roles = build_roles(scene, options, prior_logdim, prior_sigma)
    relations = Relationships(scene.ev, roles, options.relation_weight)
    active = [r for r in roles if r is not None]
    old = scene.ev.old_pca.astype(float)
    steps, calls, trace, trace_boxes = 0, 0, [], []
    starts = [[h.box.copy() for h in r.hypotheses] for r in active]

    def selected_box(selection):
        return old.copy() if not selection or selection[0] < 0 else roles[0].hypotheses[selection[0]].box.copy()

    def record(iteration, shift, moved):
        energy = relations.energy()
        for r in active:
            energy += support_energy(r, r.state, preparation(r, [h.box for h in r.hypotheses]))[0]
            energy += sum(h.fit_weight * h.fitter.regularize(to_parameters(h.box))[0] for h in r.hypotheses)
        chosen = choose(scene, roles, relations)
        trace.append((iteration, energy, shift, moved, int(chosen[0]) if chosen else -1))
        trace_boxes.append(selected_box(chosen))
        return chosen

    selected = record(0, 0., 0.)
    rounds, sweeps = (1, 4) if options.path == 'one' else (3, 4)
    if options.path == 'serial':
        for r, boxes in zip(active, starts):
            steps += improve_support(r, boxes, rounds * sweeps)
    for iteration in range(1, rounds + 1):
        prior_states = [r.state.copy() for r in active]
        prior_boxes = [[h.box.copy() for h in r.hypotheses] for r in active]
        if options.path != 'serial':
            fitted = starts if options.path == 'no_feedback' else prior_boxes
            for r, boxes in zip(active, fitted):
                steps += improve_support(r, boxes, sweeps)
        calls += fit_roles(roles, relations)
        change = sum(np.abs(r.state - prev).sum() for r, prev in zip(active, prior_states))
        changed = sum(np.linalg.norm(h.box[:3] - old[:3]) for r, boxes in zip(active, prior_boxes)
                      for h, old in zip(r.hypotheses, boxes))
        selected = record(iteration, float(change), float(changed))
    n = len(roles)
    s, q = np.zeros((n, scene.nseed), np.float32), np.zeros((n, scene.nseed), np.float32)
    role_boxes = np.full((n, 7), np.nan)
    queries = np.full(n, -2, int)  # -2 UNKNOWN, -1 scene-generated, >=0 old query.
    for e, (r, j) in enumerate(zip(roles, selected)):
        if r is None or j < 0:
            continue
        s[e], q[e] = r.state[:scene.nseed, j], r.prior[:scene.nseed, j]
        role_boxes[e] = r.hypotheses[j].box
        queries[e] = r.hypotheses[j].query
    subject = roles[0] if roles else None
    region_list = subject.hypotheses if subject else []
    h = np.array([a.box for a in region_list]).reshape(-1, 7)
    origins = np.array([a.query for a in region_list], dtype=np.int16)
    fallback = not selected or selected[0] < 0
    old_support = np.zeros(scene.nseed, np.float32)
    if subject:
        for j, r in enumerate(region_list):
            if r.query == scene.ev.old_winner:
                old_support = subject.state[:scene.nseed, j].astype(np.float32)
                break
    return Result(selected_box(selected), h, origins, s, q, role_boxes, queries, int(queries[0]) if n else -2,
                  old_support, calls, steps, fallback, trace, np.stack(trace_boxes), time.perf_counter() - tic)


def simple_boxes(scene, log_dims, sigma, weights=None):
    """All geometry controls reuse one old winner and the same robust box fitter."""
    old = scene.ev.old_pca.astype(float)
    raw = of_query_box(scene, scene.ev.old_winner)
    ix = scene.select_near(old, locked=raw)
    crop = scene.points[ix]
    pts = dict(pca_obb=pca_quantile_box(crop, old))
    for name in ('uniform_fit', 'uniform_fixed'):
        fit = BoxFitter(old, log_dims, sigma, fixed_dims=(name == 'uniform_fixed'))
        value = old
        for _ in range(3):
            value, _ = fit.fit(crop, np.ones(len(crop)), value)
        pts[name] = value
    if weights is not None:
        ix = scene.select_near(old)
        local = scene.points[ix]
        w = (weights[scene.neighbor[ix]] * scene.interpolation[ix]).sum(1)
        fit = BoxFitter(old, log_dims, sigma)
        value = old
        for _ in range(3):
            value, _ = fit.fit(local, w, value)
        pts['fixed_winner_support'] = value
    return pts
