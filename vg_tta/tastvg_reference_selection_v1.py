"""Fixed candidate qualification: frame acquisition, not a new online update."""
import numpy as np


def student_frames(candidates, frame_ids):
    ids = np.asarray(frame_ids)
    spans = np.asarray([c['physical_interval'] for c in candidates], float)
    assert len(ids) >= 5 and len(spans) > 0 and (np.diff(ids) > 0).all()
    w = ((ids[None, :] >= spans[:, :1]) & (ids[None, :] < spans[:, 1:])).mean(0)
    assert w.sum() > 0
    cdf = w.cumsum()/w.sum()
    raw = np.searchsorted(cdf, [.1, .3, .5, .7, .9], side='left').tolist()
    selected = set(raw)
    middle = cdf-w/(2*w.sum())
    filled = []
    while len(selected) < 5:
        available = [i for i in range(len(ids)) if i not in selected and w[i] > 0]
        if not available:
            available = [i for i in range(len(ids)) if i not in selected]
        # Prefer unobserved positive support; maximize CDF separation, then
        # support weight, then earliest sample. No GT enters this rule.
        i = max(available, key=lambda j: (min(abs(middle[j]-middle[k]) for k in selected), w[j], -j))
        selected.add(i); filled.append(i)
    return dict(weights=w.tolist(), raw_quantiles=raw, raw_unique=len(set(raw)),
                positions=sorted(selected), filled=filled, actual_unique=len(selected))


def decision(candidate_boxes, expert_boxes, valid):
    from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
    r = rewards(candidate_boxes, expert_boxes, valid)
    return dict(available=r is not None, rewards=None if r is None else r.tolist(),
                selected=0 if r is None else int(np.argmax(r)),
                empty_fallback='native' if r is None else None)


def pairwise(rewards, utilities, eps=1e-12):
    """All 36 pairs; empty feedback and expert ties earn .5 on GT-strict pairs."""
    u = np.asarray(utilities, float)
    assert len(u) == 9
    r = np.zeros(9) if rewards is None else np.asarray(rewards, float)
    pairs = []; correct = decisive = count = gt_ties = 0
    for i in range(9):
        for j in range(i+1, 9):
            d, g = float(r[i]-r[j]), float(u[i]-u[j])
            a, b = int(d > eps)-int(d < -eps), int(g > eps)-int(g < -eps)
            score = None if b == 0 else .5 if a == 0 else float(a == b)
            pairs.append(dict(i=i, j=j, reward_difference=d, utility_difference=g,
                              expert_sign=a, GT_sign=b, accuracy=score))
            gt_ties += b == 0
            if b:
                count += 1; correct += score; decisive += a != 0
    return dict(pairs=pairs, strict_GT_pairs=count, GT_ties=gt_ties,
                decisive_pairs=decisive, pairwise_accuracy=correct/count if count else None,
                decisive_coverage=decisive/count if count else None)
