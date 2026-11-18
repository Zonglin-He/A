"""Geometry-constrained point, image and cross-modal supports for LCIB."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from geometry162 import axes, in_box, local_points


@dataclass
class SeedHypothesis:
    query: int
    box: np.ndarray
    seed_mask: np.ndarray
    log_prior: float


def uniform_valid(index, weights):
    weights = np.where(index >= 0, weights, 0.)
    total = weights.sum(1, keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights, dtype=float), where=total > 0)


def edge_block(indices, weights, ncol):
    valid = (indices >= 0) & (weights > 0)
    source, slot = np.nonzero(valid)
    return sparse.csr_matrix((weights[source, slot], (source, indices[source, slot])),
                             shape=(len(indices), ncol))


def neighbor_graph(seeds):
    tree = cKDTree(seeds)
    k = min(7, len(seeds))
    distance, neighbors = tree.query(seeds, k=k)
    scale = np.clip(distance[:, min(k - 1, 3)], .12, 3.)
    near = np.minimum(scale[:, None], scale[neighbors])
    weight = np.exp(-.5 * (distance / near) ** 2)
    valid = (distance <= 1.9 * near) & (neighbors != np.arange(len(seeds))[:, None])
    raw = edge_block(neighbors, uniform_valid(neighbors, weight * valid), len(seeds))
    return (raw + raw.T) * .5, tree, scale


def image_graph(features, supported):
    height, width = 32, 48
    rows, cols = np.indices((height, width))
    ix = (rows * width + cols).reshape(-1)
    origin = np.r_[ix[cols.ravel() < width - 1], ix[rows.ravel() < height - 1]]
    dest = origin + np.r_[np.ones((width - 1) * height, int), np.full((height - 1) * width, width, int)]
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    cosine = (normalized[origin] * normalized[dest]).sum(1)
    weight = np.exp(-(1 - np.clip(cosine, -1, 1)) / .25)
    weight *= supported[origin] | supported[dest]
    w = sparse.csr_matrix((weight / 4, (origin, dest)), shape=(len(ix), len(ix)))
    return w + w.T


class SceneSupport:
    def __init__(self, ev):
        self.ev = ev
        self.seeds = ev.seeds.astype(float)
        self.nseed = len(self.seeds)
        self.nimage = len(ev.ionly)
        self.geometry, tree, intervals = neighbor_graph(self.seeds)
        # Repeated sampling of a raw point is one observation.
        unique = np.sort(np.unique(ev.indices, return_index=True)[1])
        self.points = ev.points[unique].astype(float)
        self.unique = unique
        d, n = tree.query(self.points, k=3)
        valid = d <= np.maximum(.30, 2.25 * intervals[n])
        w = valid / np.maximum(d, .04) ** 2
        self.neighbor = n
        self.interpolation = uniform_valid(n, w)
        self.box_support = (
            np.abs(self.seeds[None] - ev.boxes[:, None, :3]) <= ev.boxes[:, None, 3:6] / 2
        ).all(-1)
        self.swap = self.adjacent_instance_swap()
        self.graphs = {}

    def adjacent_instance_swap(self):
        """Exchange equal nonoverlapping seed sets in two nearby candidate regions."""
        ev = self.ev
        perm = np.arange(self.nseed)
        winner = ev.old_winner
        first = self.box_support[winner] & ev.visible
        if first.sum() < 2:
            return perm
        scores = ev.utility.copy()
        candidates = np.argsort(-scores, kind='stable')[:32]
        center = ev.boxes[winner, :3]
        d = np.linalg.norm(ev.boxes[candidates, :3] - center, axis=1)
        for q in candidates[np.argsort(d, kind='stable')]:
            other = self.box_support[q] & ev.visible
            if (first & other).any() or other.sum() < 2 or d[candidates == q][0] > 2.5 * np.linalg.norm(ev.old_pca[3:6]):
                continue
            left, right = np.flatnonzero(first), np.flatnonzero(other)
            n = min(len(left), len(right))
            left = left[np.argsort(np.linalg.norm(self.seeds[left] - center, axis=1))[:n]]
            right = right[np.argsort(np.linalg.norm(self.seeds[right] - ev.boxes[q, :3], axis=1))[:n]]
            perm[left], perm[right] = right, left
            return perm
        return perm

    def graph(self, kind):
        """Keep native A and C as two measurements, then add their quadratic energies."""
        if kind in self.graphs:
            return self.graphs[kind]
        ev = self.ev
        ps, im = self.nseed, self.nimage
        if kind == 'no_rgb_graph':
            A = sparse.csr_matrix((ps, im))
            C = A.T
            local = sparse.csr_matrix((im, im))
            imaging = np.zeros(im, dtype=bool)
        else:
            ip, pi = ev.p2i, ev.i2p
            if kind == 'uniform_cross':
                aw = uniform_valid(ip, ip >= 0)
                cw = uniform_valid(pi, pi >= 0)
            else:
                aw = ev.p2i_learned
                cw = ev.i2p_learned
            if kind == 'swap_cross':
                ip, aw = ip[self.swap], aw[self.swap]
                pi = np.where(pi >= 0, self.swap[np.maximum(pi, 0)], -1)
            A, C = edge_block(ip, aw, im), edge_block(pi, cw, ps)
            imaging = (np.asarray(A.sum(0)).ravel() + np.asarray(C.sum(1)).ravel()) > 0
            local = image_graph(ev.ionly, imaging)
        bridges = (A + C.T) * .42
        graph = sparse.bmat([[self.geometry * .62, bridges], [bridges.T, local * .32]], format='csr')
        degree = np.asarray(graph.sum(1)).ravel()
        self.graphs[kind] = graph, degree, C, imaging
        return self.graphs[kind]

    def select_near(self, box, locked=None, max_points=320):
        if locked is None:
            local = local_points(self.points, box)
            include = (np.abs(local) <= np.maximum(.88 * box[3:6], .20)).all(1)
        else:
            include = in_box(self.points, locked)
        ids = np.flatnonzero(include)
        if len(ids) > max_points:
            ids = ids[np.linspace(0, len(ids) - 1, max_points, dtype=int)]
        return ids

    def candidates(self, role, prior, mode):
        """Old regions initialize evidence; separated full-field peaks add physical alternatives."""
        ev = self.ev
        att = ev.affinity[role].astype(float)
        masks = self.box_support
        count = masks.sum(1)
        mass = (masks @ att) / np.sqrt(np.maximum(count, 1))
        normalized = mass / max(mass.max(), 1e-12)
        if role == 0:
            old_rank = ev.utility / ev.utility.max()
            priority = (.35 + .65 * normalized) * old_rank
            order = [ev.old_winner, *np.argsort(-priority, kind='stable').tolist()]
        else:
            old_rank = normalized
            routed = [int(q) for q in ev.hypotheses[role] if q >= 0]
            order = [*routed, *np.argsort(-normalized, kind='stable').tolist()]
        result = []
        for q in order:
            raw = ev.boxes[q]
            if np.any(raw[3:6] <= .12) or (count[q] < 2 and q != ev.old_winner):
                continue
            if q in (h.query for h in result):
                continue
            if self.duplicates(result, raw[:3], raw[3:6], masks[q]):
                continue
            init = ev.old_pca.astype(float) if q == ev.old_winner else self.orient(raw)
            result.append(SeedHypothesis(q, init, masks[q], np.log(max(old_rank[q], .025))))
            if len(result) == 4:
                break
        if mode != 'scene':
            return result
        prior_dim = np.exp(prior)
        connected = self.geometry
        den = np.asarray(connected.sum(1)).ravel()
        smooth = (att + connected @ att) / (1 + den)
        cap = len(result) + 2
        for peak in np.argsort(-smooth, kind='stable')[:64]:
            if smooth[peak] < 1.25 / len(att):
                break
            root = self.seeds[peak]
            nearest = ((ev.boxes[:, :3] - root) ** 2).sum(1).argsort(kind='stable')
            match = next((q for q in nearest if np.all(ev.boxes[q, 3:6] > .20)), None)
            dims = prior_dim.copy()
            if match is not None and np.linalg.norm(ev.boxes[match, :3] - root) < np.linalg.norm(prior_dim):
                dims = ev.boxes[match, 3:6].astype(float)
            within = (np.abs(self.seeds - root) < .7 * np.maximum(dims, .65)).all(1)
            allowed = within & (smooth >= .55 * smooth[peak])
            region = np.zeros(self.nseed, dtype=bool)
            region[peak] = True
            waiting = [peak]
            while waiting:
                j = waiting.pop()
                near = connected.indices[connected.indptr[j]:connected.indptr[j+1]]
                new = near[allowed[near] & ~region[near]]
                region[new] = True
                waiting.extend(new.tolist())
            center = np.average(self.seeds[region], axis=0, weights=smooth[region])
            if self.duplicates(result, center, dims, region) or len(self.select_near(np.r_[center, dims, 0.])) < 8:
                continue
            value = .55 * float(smooth[peak] / max(smooth.max(), 1e-12))
            result.append(SeedHypothesis(-1, self.orient(np.r_[center, dims]), region, np.log(max(.025, value))))
            if len(result) >= cap:
                break
        return result

    def orient(self, raw):
        q = np.r_[raw, 0.]
        ix = self.select_near(q, locked=q)
        if len(ix) < 8 or raw[3] < 1.5 * raw[4]:
            return q
        _, v = np.linalg.eigh(np.cov(self.points[ix, :2], rowvar=False))
        q[6] = (np.arctan2(v[1, 1], v[0, 1]) + np.pi / 2) % np.pi - np.pi / 2
        return q

    @staticmethod
    def duplicates(existing, center, dims, mask):
        for region in existing:
            shared = (region.seed_mask & mask).sum()
            union = (region.seed_mask | mask).sum()
            if shared > .55 * max(1, union):
                return True
            if np.linalg.norm((center - region.box[:3]) / np.maximum(np.minimum(dims, region.box[3:6]), .3)) < .40:
                return True
        return False

    def evidence(self, role, hypotheses, options):
        _, _, reverse, _ = self.graph(options.graph)
        N = self.nseed
        affinity = np.log(N * self.ev.affinity[role] + 1e-6) / options.temperature - options.role_bias
        if options.missing_negative:
            affinity = affinity - 2. * (~self.ev.visible)
        prior = np.asarray([h.log_prior for h in hypotheses])
        unary = []
        keep = []
        for h in hypotheses:
            difference = local_points(self.seeds, h.box) / np.maximum(.65 * h.box[3:6], .25)
            distance = np.square(difference).sum(1)
            hint = .7 * h.seed_mask
            unary.append(-.5 * distance + hint)
            keep.append(h.seed_mask if options.init == 'inside' else (distance <= 12.0))
        allowed = np.c_[np.stack(keep, axis=1), np.ones(N, bool)]
        log_q = np.stack(unary, axis=1) + affinity[:, None] + prior[None] - logsumexp(prior)
        log_q = np.c_[log_q, np.zeros(N)]
        log_q[~allowed] = -np.inf
        q = np.exp(log_q - logsumexp(log_q, axis=1, keepdims=True))
        iq = reverse @ q
        total = iq.sum(1, keepdims=True)
        iq = np.divide(iq, total, out=iq, where=total > 0) * .85
        iq[:, -1] = 1 - iq[:, :-1].sum(1)
        return np.vstack((q, iq)), np.vstack((allowed, np.ones_like(iq, bool)))
