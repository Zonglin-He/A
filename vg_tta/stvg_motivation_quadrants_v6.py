"""Source-independent native T/S definitions; no labels or model on import."""
import numpy as np

CATEGORY_NAMES = ('T+/S+', 'T+/S-', 'T-/S+', 'T-/S-')


def quadrant(t, s, threshold=.5):
    t = np.asarray(t, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    assert t.shape == s.shape and t.ndim == 1 and len(t)
    assert np.isfinite(t).all() and np.isfinite(s).all()
    assert ((t >= 0) & (t <= 1 + 1e-12)).all() and ((s >= 0) & (s <= 1 + 1e-12)).all()
    return np.where(t > threshold, np.where(s > threshold, 0, 1), np.where(s > threshold, 2, 3))


def summarize_models(values, seed=20261009, draws=10000, threshold=.5):
    names = list(values)
    categories = {m: quadrant(*values[m], threshold) for m in names}
    n = len(categories[names[0]])
    assert all(len(x) == n for x in categories.values())
    rng = np.random.default_rng(seed)
    boots = {m: [] for m in names}
    for start in range(0, draws, 250):
        ids = rng.integers(0, n, size=(min(250, draws-start), n))
        for model in names:
            boots[model].append(np.stack([(categories[model][ids] == j).mean(1) * 100 for j in range(4)], axis=1))
    boot = {m: np.concatenate(v, axis=0) for m, v in boots.items()}
    result = {}
    for model in names:
        counts = [int((categories[model] == j).sum()) for j in range(4)]
        pct = np.array(counts) / n * 100
        contrast = boot[model][:, 1] - boot[model][:, 2]
        result[model] = dict(N=n, categories=list(CATEGORY_NAMES), counts=counts,
            percent=pct.tolist(), ci95_percent=np.quantile(boot[model], [.025, .975], axis=0).T.tolist(),
            contrast_Tplus_Sminus_minus_Tminus_Splus_pp=float(pct[1]-pct[2]),
            contrast_ci95_pp=np.quantile(contrast, [.025, .975]).tolist(),
            threshold=threshold, strict_greater_than=True)
    between = {}
    for a, b in zip(names, names[1:]):
        delta = boot[a]-boot[b]
        between[a+' minus '+b] = dict(percent_difference=(np.array(result[a]['percent'])-np.array(result[b]['percent'])).tolist(),
            ci95_paired_parent_bootstrap=np.quantile(delta, [.025, .975], axis=0).T.tolist())
    return dict(models=result, paired_between_backbones=between, seed=seed, bootstrap_draws=draws,
        parent_source_unit=True, thresholds_selected_from_results=False)


def independent_components(boxes, ids, interval, truth, span, width, height, hc2=False,
        temporal_valid=True, spatial_valid=True):
    """GT-fixed sIoU, anchor-hull support, physical half-open temporal arithmetic."""
    ids = np.asarray(ids, dtype=int)
    dense = np.array(sorted(truth), dtype=int)
    gt = np.array([truth[int(i)] for i in dense], dtype=np.float64)
    assert gt.shape == (len(dense), 4) and len(dense)
    assert np.isfinite(gt).all() and (gt[:, 2:] >= gt[:, :2]).all()
    if temporal_valid:
        assert interval is not None and len(interval) == 2
        a, b = map(int, interval)
    else:
        a, b = 0, 0
    g, h = map(int, span)
    overlap = max(0, min(b, h)-max(a, g)) if b > a else 0
    denom = max(b-a, 0) + max(h-g, 0) - overlap
    t = overlap/denom if denom > 0 else 0.
    all_iou = np.zeros(len(dense))
    covered = np.zeros(len(dense), dtype=bool)
    invalid = 0
    raw = np.asarray(boxes, dtype=np.float64)
    if len(ids) and spatial_valid:
        assert ids.tolist() == sorted(set(ids.tolist()))
        assert raw.shape == (len(ids), 4) and np.isfinite(raw).all()
        pix = raw*np.array([width, height, width, height])
        if hc2:
            pix = np.maximum(pix, 0)  # literal existing HC2 official convention
        interp = np.column_stack([np.interp(dense, ids, pix[:, j]) for j in range(4)])
        covered = (dense >= ids[0]) & (dense <= ids[-1])
        geometry = (interp[:, 2:] > interp[:, :2]).all(1)
        invalid = int((covered & ~geometry).sum())
        intersection = np.maximum(np.minimum(interp[:, 2:], gt[:, 2:])-np.maximum(interp[:, :2], gt[:, :2]), 0).prod(1)
        union = np.maximum(interp[:, 2:]-interp[:, :2], 0).prod(1) + np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1) - intersection
        valid = covered & geometry & (union > 0)
        all_iou[valid] = intersection[valid]/union[valid]
    else:
        pix = np.zeros((0, 4))
    return dict(tIoU=float(t), sIoU=float(all_iou.mean()), spatial_coverage=float(covered.mean()),
        uncovered_GT_frames=int((~covered).sum()), invalid_GT_box_frames=invalid,
        GT_frames=len(dense), interval=[a, b], pixel_boxes=pix, GT_frame_ids=dense,
        per_GT_frame_IoU=all_iou)
