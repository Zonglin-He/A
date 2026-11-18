"""Candidate retrieval and matched exact readouts for direct camera relations."""
import numpy as np
from scipy.special import logsumexp

from scope167 import ARMS, BANK, NOVEL, NROLE
from crossplatform.ground_single_frame import orient_box
from geometry162 import in_box


def direct_edges(edges):
    return tuple(dict.fromkeys((int(b), int(sign)) if a == 0 else (int(a), -int(sign))
                               for a, b, sign in edges if (a == 0) != (b == 0)))


def append_sealed_regions(raw, old):
    regions = []
    for group, (size, _, novel) in zip(old['starts'], old['budget'], strict=True):
        if novel:
            regions.extend(group[int(size - novel):int(size)])
    assert len(regions) <= NOVEL
    padded = np.full((NOVEL, 7), np.nan)
    if regions:
        padded[:len(regions)] = regions
    return np.concatenate((np.c_[raw, np.zeros(len(raw))], padded[:len(regions)])), padded


def support_matrix(geometry, seeds, native):
    axis = (np.abs(seeds[None] - geometry[:native, None, :3]) <= geometry[:native, None, 3:6] / 2).all(-1)
    if len(geometry) == native:
        return axis
    novel = np.stack([in_box(seeds, b) for b in geometry[native:]])
    return np.concatenate((axis, novel))


def support_overlap(masks):
    values = masks.astype(np.float32)
    hits = values @ values.T
    size = masks.sum(1)
    union = size[:, None] + size[None, :] - hits
    overlap = np.divide(hits, union, out=np.zeros_like(union), where=union > 0)
    return hits, overlap


def ranked_distinct(score, allowed, geometry, overlap):
    order = np.argsort(-score, kind='stable')
    usable = allowed.copy()
    selected = []
    for i in order:
        if not usable[i]:
            continue
        selected.append(int(i))
        if len(selected) == BANK:
            break
        delta = (geometry[:, :3] - geometry[i, :3]) / np.maximum(np.minimum(geometry[:, 3:6], geometry[i, 3:6]), .30)
        usable &= (overlap[i] <= .50) & ((delta * delta).sum(1) >= .16)
    return np.array(selected, dtype=np.int16)


def transfer(values, hits, overlap, native):
    """Discount borrowed semantic evidence based on overlap with native support."""
    affinity = np.where(hits[native:, :native] >= 2, overlap[native:, :native], 0.)
    inherited = np.max(np.exp(values)[None, :] * affinity, axis=1, initial=0.)
    return np.r_[values, np.log(np.maximum(inherited, .025))]


def support_intervals(masks, visible, uv):
    lower, upper, confidence = np.zeros((3, len(masks)))
    good = masks & visible[None, :]
    raster = np.floor(uv[visible] * np.array([48, 32])).astype(np.int32)
    cells = np.full(len(uv), -1)
    cells[visible] = raster[:, 0] + 48 * raster[:, 1]
    for i in np.flatnonzero(good.sum(1) >= 3):
        lower[i], upper[i] = np.quantile(uv[good[i], 0], [.1, .9])
        n = len(np.unique(cells[good[i]]))
        confidence[i] = n / (n + 8.)
    return lower, upper, confidence


def components(subjects, references, lower, upper, confidence, overlap):
    left = np.zeros((len(subjects), len(references)))
    cost = left.copy()
    real = references >= 0
    if real.any():
        ix = np.ix_(subjects, references[real])
        center, width = (lower + upper) / 2, upper - lower
        delta = center[references[real]][None, :] - center[subjects][:, None]
        uncertainty = np.maximum(.02, (width[subjects][:, None] + width[references[real]][None, :]) / 2)
        known = np.sqrt(confidence[subjects][:, None] * confidence[references[real]][None, :])
        left[:, real] = known * np.tanh(delta / uncertainty)
        cost[:, real] = np.maximum(0., (overlap[ix] - .35) / .65)
    return left, cost


def select(subjects, refs, scores, factors, signs, mode):
    """Optimize a star whose edges join a single subject to each reference role."""
    decision = np.full(NROLE, -1, dtype=np.int16)
    subject_score = scores[0][subjects].copy()
    reference_positions = {}
    for role, candidates in refs.items():
        u = np.r_[scores[role][candidates[:-1]], -2.] if len(candidates) > 1 else np.zeros(1)
        left, penalty = factors[role]
        pairs = u[None, :] + signs[role] * left - penalty
        if mode == 'joint':
            pos = pairs.argmax(1)
            subject_score += pairs[np.arange(len(subjects)), pos]
            reference_positions[role] = pos
        else:
            pos = int(u.argmax())
            if mode == 'fixed_reference':
                subject_score += pairs[:, pos]
            reference_positions[role] = np.full(len(subjects), pos, dtype=int)
    i = int(subject_score.argmax())
    decision[0] = subjects[i]
    for role, pos in reference_positions.items():
        decision[role] = refs[role][pos[i]]
    return decision


def blank(base, winner):
    scores = np.full((NROLE, BANK + 1), -np.inf)
    scores[0, 0] = 0.
    bank = np.full_like(scores, -2, dtype=np.int16)
    bank[0, 0] = winner
    winners = np.full((len(ARMS), NROLE), -1, dtype=np.int16)
    winners[:, 0] = winner
    boxes = np.full((*winners.shape, 7), np.nan)
    boxes[:, 0] = base
    pair = np.zeros((NROLE - 1, BANK, BANK + 1), dtype=np.float64)
    return dict(boxes=boxes, flip_boxes=boxes.copy(), winners=winners, flip_winners=winners.copy(),
                bank=bank, score=scores, left=pair, overlap=pair.copy(), flip=False, active=False,
                novel=np.full((NOVEL, 7), np.nan), satisfied=np.zeros((2, len(ARMS), NROLE), dtype=np.int8))


def repaired(ev, old, query, tokens):
    edges = direct_edges(ev.edges)
    assert edges and len(ev.boxes) == len(query) and ev.role_mask.shape[1] == len(tokens)
    native = len(ev.boxes)
    geometry, novel = append_sealed_regions(ev.boxes, old)
    keep = np.sort(np.unique(ev.seeds, axis=0, return_index=True)[1])
    masks = support_matrix(geometry, ev.seeds[keep], native)
    hits, overlap = support_overlap(masks)
    count = masks.sum(1)
    allowed = (count >= 2) & (geometry[:, 3:6] > .12).all(1)
    scores = np.zeros((NROLE, len(geometry)))
    scores[0] = transfer(np.log(np.maximum(ev.utility / ev.utility.max(), 1e-9)), hits, overlap, native)
    allowed_subject = allowed.copy()
    allowed_subject[ev.old_winner] = True
    subjects = ranked_distinct(scores[0], allowed_subject, geometry, overlap)
    assert subjects[0] == ev.old_winner
    reference_roles = sorted({r for r, _ in edges})
    logits = query.astype(float) @ tokens.astype(float).T / .07
    lower, upper, confidence = support_intervals(masks, ev.visible[keep], ev.uv[keep])
    refs, legacy, shared, restricted = {}, {}, {}, {}
    out = blank(ev.old_pca, ev.old_winner)
    out['novel'] = novel
    out['bank'][0, :len(subjects)] = subjects
    out['score'][0, :len(subjects)] = scores[0, subjects]
    for r in reference_roles:
        if ev.role_mask[r].any() and allowed.any():
            words = logits[:, ev.role_mask[r]]
            match = logsumexp(words, axis=1) - np.log(words.shape[1])
            match = transfer(match - match.max(), hits, overlap, native)
            density = (masks @ ev.affinity[r, keep]) / np.sqrt(np.maximum(count, 1))
            density = np.log(np.maximum(density / max(float(density[allowed].max()), 1e-12), .025))
            scores[r] = .5 * (match + density)
            scores[r] -= scores[r, allowed].max()
            pool = ranked_distinct(scores[r], allowed, geometry, overlap)
            old_pool = [int(q) for q in old['routes'][r] if q >= 0 and allowed[q]]
        else:
            pool, old_pool = (), ()
        refs[r] = np.array([*pool, -1], dtype=np.int16)
        legacy[r] = np.array([*old_pool, -1], dtype=np.int16)
        shared[r] = components(subjects, refs[r], lower, upper, confidence, overlap)
        restricted[r] = components(subjects, legacy[r], lower, upper, confidence, overlap)
        n = len(refs[r])
        out['bank'][r, :n] = refs[r]
        out['score'][r, :n] = np.r_[scores[r, refs[r][:-1]], -2.] if n > 1 else np.zeros(1)
        out['left'][r - 1, :len(subjects), :n] = shared[r][0]
        out['overlap'][r - 1, :len(subjects), :n] = shared[r][1]

    signs = {r: sum(v for role, v in edges if role == r) for r in reference_roles}
    out['flip'] = len(edges) == 1
    out['active'] = any(signs[r] and np.any(shared[r][0]) for r in reference_roles)
    oriented = {ev.old_winner: ev.old_pca}
    for f, key in enumerate(('winners', 'flip_winners')):
        if f and not out['flip']:
            out[key] = out['winners'].copy()
            out['flip_boxes'] = out['boxes'].copy()
            out['satisfied'][f] = out['satisfied'][0]
            break
        side = {r: -val if f else val for r, val in signs.items()}
        for i, arm in enumerate(ARMS[1:], 1):
            pool, factor = (legacy, restricted) if arm == 'r88_ref3_joint' else (refs, shared)
            choice = select(subjects, pool, scores, factor, side,
                            'joint' if arm == 'r88_ref3_joint' else arm)
            out[key][i] = choice
            choice_boxes = out['flip_boxes' if f else 'boxes'][i]
            for r in (0, *reference_roles):
                q = int(choice[r])
                if q >= 0:
                    if q not in oriented:
                        oriented[q] = orient_box(ev.points, geometry[q, :6]) if q < native else geometry[q]
                    choice_boxes[r] = oriented[q]
                else:
                    choice_boxes[r] = np.nan
                if r and q >= 0 and side[r]:
                    a = np.flatnonzero(subjects == choice[0])[0]
                    b = np.flatnonzero(pool[r] == q)[0]
                    out['satisfied'][f, i, r] = np.sign(side[r] * factor[r][0][a, b])
    return out
