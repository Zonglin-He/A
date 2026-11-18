"""Exact fixed-bank replay with an explicitly reachable selective-binding state."""
import hashlib

import numpy as np
from scipy.special import logsumexp

from scope168 import ARMS
from retrieval167 import support_matrix


def role_terms(raw, novel, seeds, query, tokens, masks, valid, bank, old_score):
    """Recover the frozen attention-plus-native score after replacing its native term."""
    logits = query.astype(float) @ tokens.astype(float).T / .07
    denominator = logsumexp(logits[:, valid], axis=1)
    native = len(raw)
    extras = novel[np.isfinite(novel).all(1)]
    if len(extras):
        geometry = np.concatenate((np.c_[raw, np.zeros(native)], extras))
        support = support_matrix(geometry, np.unique(seeds, axis=0), native).astype(np.float32)
        hits = support[native:] @ support[:native].T
        size = support.sum(1).astype(np.int64)
        union = size[native:, None] + size[None, :native] - hits
        weights = np.divide(hits, union, where=(hits >= 2), out=np.zeros_like(union))
    else:
        weights = np.zeros((0, native))

    def transfer(x, floor):
        return np.r_[x, np.log(np.maximum(np.max(weights * np.exp(x)[None], axis=1), floor))]

    new = old_score.copy()
    change = np.zeros_like(old_score)
    features = np.full((len(bank), 2), np.nan)
    for role in range(1, len(bank)):
        physical = bank[role] >= 0
        if not physical.any():
            continue
        use = masks[role]
        nwords = use.sum()
        assert nwords and not (use & ~valid).any()
        numerator = logsumexp(logits[:, use], axis=1)
        logmass = numerator - denominator
        bounded = transfer(logmass - logmass.max(), .025)
        current = transfer(numerator - numerator.max(), .025)
        correction = .5 * (bounded - current)[bank[role, physical]]
        change[role, physical] = correction
        change_score = old_score[role, physical] + correction
        new[role, physical] = change_score - change_score.max()
        old = old_score[role, physical]
        first = bank[role, physical][int(old.argmax())]
        margin = np.sort(old)[-1] - np.sort(old)[-2] if len(old) > 1 else 0.
        actual = transfer(logmass, 1e-12)[first]
        features[role] = (actual - np.log(nwords / valid.sum()), margin)
    return new, change, features


def scramble(old, change, bank, key):
    output = old.copy()
    count = 0
    for r in range(1, len(bank)):
        physical = bank[r] >= 0
        n = int(physical.sum())
        if n < 2:
            continue
        code = hashlib.sha256(f'R168|{key}|{r}'.encode()).digest()
        shift = 1 + int.from_bytes(code[:4], 'little') % (n - 1)
        delta = change[r, physical]
        moved = np.roll(delta, shift)
        count += int(not np.array_equal(delta, moved))
        scores = old[r, physical] + moved
        output[r, physical] = scores - scores.max()
    return output, count


def gate_logits(feature, rule):
    out = np.full(len(feature), np.nan)
    good = np.isfinite(feature).all(1)
    x = (feature[good] - np.array(rule['mean'])) / np.array(rule['scale'])
    x = np.clip(x, rule['lo'], rule['hi'])
    beta = np.asarray(rule['coef'])
    out[good] = beta[0] + x @ beta[1:]
    return out


def selective(score, bank, logits):
    out = score.copy()
    for r in range(1, len(bank)):
        physical = bank[r] >= 0
        null = bank[r] == -1
        if not null.any():
            continue
        out[r, null] = 0.
        if physical.any():
            assert np.isfinite(logits[r])
            out[r, physical] = (score[r, physical] - score[r, physical].max() + logits[r]
                                if logits[r] > 0. else -np.inf)
    return out


def direct_signs(direct, swap=False):
    return {r: (-1 if swap else 1) * sum(s for k, s in direct if k == r) for r in sorted({r for r, _ in direct})}


def solve(bank, score, left, overlap, signs, motion=True, overlap_on=True, independent=False):
    """Return the assignment, exact decomposition, and the best terms at the original subject."""
    subjects = np.flatnonzero(bank[0] >= 0)
    n = len(subjects)
    terms = np.zeros((n, 4), dtype=float)
    terms[:, 0] = score[0, subjects]
    positions = {}
    ii = np.arange(n)
    for r, sign in signs.items():
        world = np.flatnonzero(bank[r] >= -1)
        cu = score[r, world]
        relation = left[r - 1][np.ix_(subjects, world)] * (sign if motion and not independent else 0)
        penalty = -overlap[r - 1][np.ix_(subjects, world)] if overlap_on and not independent else np.zeros_like(relation)
        z = cu[None, :] + relation + penalty
        pick = z.argmax(1)
        positions[r] = world[pick]
        terms[:, 1] += cu[pick]
        terms[:, 2] += relation[ii, pick]
        terms[:, 3] += penalty[ii, pick]
    # Accumulate the score in the same order as the R167 implementation.
    total = score[0, subjects].copy()
    for r, sign in signs.items():
        p = positions[r]
        relation = left[r - 1, subjects, p] * (sign if motion and not independent else 0)
        overlap_penalty = overlap[r - 1, subjects, p] if overlap_on and not independent else 0
        total += (score[r, p] + relation - overlap_penalty)
    i = int((score[0, subjects] if independent else total).argmax())
    start = int(score[0, subjects].argmax())
    result = np.full(len(bank), -1, dtype=np.int16)
    result[0] = bank[0, subjects[i]]
    for r, p in positions.items():
        result[r] = bank[r, p[i]]
    diff = terms[i] - terms[start]
    assert abs(diff.sum() - (total[i] - total[start])) <= 1e-10
    assert diff.sum() >= -1e-10
    return result, diff


def policies(bank, score, left, overlap, direct, new, change, feature, rule, key, swap=False):
    signs = direct_signs(direct, swap)
    logit = gate_logits(feature, rule)
    faulty, displaced = scramble(score, change, bank, key)
    states = (score, score, score, score, selective(score, bank, logit), selective(new, bank, logit),
              selective(faulty, bank, logit), new)
    out = []
    for arm, u in zip(ARMS, states, strict=True):
        out.append(solve(bank, u, left, overlap, signs, motion=arm != 'overlap_only',
                         overlap_on=arm != 'left_only', independent=arm.endswith('independent')))
    return np.stack([c for c, _ in out]), np.stack([d for _, d in out]), logit, displaced


def legacy_availability(bank, score, left, overlap, direct):
    """Minimum physical advantage over the archived null across all allowed subjects."""
    result = []
    subject = bank[0] >= 0
    for r, sign in direct_signs(direct).items():
        phys = bank[r] >= 0
        if phys.any():
            idx = np.ix_(subject, phys)
            value = score[r, phys][None] + sign * left[r - 1][idx] - overlap[r - 1][idx]
            result.append(float(value.max(1).min() - score[r, bank[r] == -1][0]))
    return result
