"""Point support and source-calibrated pair ranking for one fixed v1 carrier."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


POINT_FEATURES = (
    'inside_mass', 'soft_enclosure', 'surface_mass', 'exterior_distance',
    'interior_depth', 'support_density', 'x_face', 'y_face', 'z_face',
    'x_span', 'y_span', 'z_span', 'x_offset', 'y_offset', 'z_offset',
)
BOX_FEATURES = ('log_length', 'log_width', 'log_height',
                'shift_x', 'shift_y', 'shift_z', 'axial_yaw_shift')
N_FEATURES = len(POINT_FEATURES) + len(BOX_FEATURES)


def axes(box):
    c, s = np.cos(box[6]), np.sin(box[6])
    return np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])


def describe(local, half, common_weight):
    relative = local / half
    absolute = np.abs(relative)
    excess = np.linalg.norm((absolute-1.).clip(0), axis=1)
    penetration = (1.-absolute.max(axis=1)).clip(0)
    inside = excess == 0.
    shell = np.exp(-.5*((excess+penetration)/.20)**2)
    enclosure = np.exp(-.5*(excess/.40)**2)
    face_distance = np.abs(absolute-1.)
    near_face = np.exp(-.5*(face_distance/.20)**2) * (excess[:,None] < .20)
    low, mid, high = np.quantile(relative, [.10,.50,.90], axis=0)
    support = np.r_[
        common_weight @ inside,
        common_weight @ enclosure,
        common_weight @ shell,
        common_weight @ excess.clip(0,2),
        common_weight @ penetration,
        np.log1p(inside.sum() / max(8.*half.prod(),.05)),
        common_weight @ near_face,
        (high-low).clip(0,8),
        np.abs(mid).clip(0,4),
    ]
    # A fixed point-fit score using the same crop, weights and box descriptors.
    direct = support[1] + .5 * support[2]
    return support, inside, direct


def pair_evidence(points, raw, refined):
    """Use one raw-anchored observation set for both candidate geometries."""
    raw, refined = (np.asarray(x,np.float64) for x in (raw,refined))
    boxes = np.stack((raw,refined))
    if boxes.shape != (2,7) or not np.isfinite(boxes).all() or (boxes[:,3:6] <= 0.).any():
        raise ValueError('Two finite oriented boxes with positive extents are required')
    half = .5 * boxes[:,3:6]
    orient = axes(raw)
    raw_local = (points-raw[:3]) @ orient
    keep = (np.abs(raw_local) <= 1.70*half[0]+.15).all(axis=1)
    points = np.unique(points[keep],axis=0)
    if not len(points):
        return np.zeros(N_FEATURES), np.zeros(3,dtype=np.int64), 0.
    raw_local = (points-raw[:3]) @ orient
    refined_local = (points-refined[:3]) @ axes(refined)
    weight = np.exp(-.5*((raw_local/(1.25*half[0]+.15))**2).sum(axis=1))
    weight /= weight.sum()
    a, inc_a, direct_a = describe(raw_local, half[0], weight)
    b, inc_b, direct_b = describe(refined_local, half[1], weight)
    delta_center = np.abs((refined[:3]-raw[:3]) @ orient)/(half[0]+.15)
    motion = np.r_[np.log(half[1]/half[0]), delta_center, abs(np.sin(refined[6]-raw[6]))]
    counts = np.asarray([inc_a.sum(), inc_b.sum(), np.count_nonzero(inc_a & inc_b)],dtype=np.int64)
    return np.r_[b-a, motion], counts, float(direct_b-direct_a)


def supported(counts):
    raw, refined, shared = np.asarray(counts).T
    return (raw>=8) & (refined>=8) & (shared>=4) & (2*shared>=np.maximum(raw,refined))


def normalized_ranking(inputs, gap, eligible, options):
    fits = eligible & (np.abs(gap)>options['tie_epsilon_iou'])
    values = inputs[fits]
    scale = np.maximum(np.sqrt(np.mean(values**2,axis=0)),.03) if len(values) else np.ones(inputs.shape[1])
    y = np.sign(gap[fits])
    w = np.minimum(np.abs(gap[fits]),options['weight_absolute_gap_cap_iou'])
    weights = np.zeros(inputs.shape[1])
    if np.unique(y).size == 2:
        w /= w.sum()
        signed = y[:,None] * values / scale
        regularization = options['regularization']

        def objective(coeff):
            margins = signed @ coeff
            loss = w @ np.logaddexp(0.,-margins) + .5*regularization*(coeff @ coeff)
            gradient = -(signed.T @ (w*expit(-margins))) + regularization*coeff
            return float(loss), gradient

        result = minimize(objective,weights,jac=True,method='L-BFGS-B',options={'gtol':1e-8, 'maxiter':200})
        if not result.success:
            raise RuntimeError(f'Source pair ranking failed to converge: {result.message}')
        weights = result.x
    return dict(rms=scale.tolist(), coefficients=weights.tolist(),
                fit_eligible=int(eligible.sum()), fit_labeled=int(len(y)),
                fit_better=int((y>0).sum()), fit_worse=int((y<0).sum()))


def margins(model, features):
    return (features / np.asarray(model['rms'])) @ np.asarray(model['coefficients'])


def rankings(scores, ious, eligible, epsilon):
    gap = ious[:,1]-ious[:,0]
    considered = eligible & (np.abs(gap)>epsilon)
    labels = gap[considered]>0.
    scores = scores[considered]
    positives = int(labels.sum())
    negatives = len(labels)-positives
    auc = None
    if positives and negatives:
        ranks = rankdata(scores,method='average')
        auc = float((ranks[labels].sum() - positives*(positives+1)/2)/(positives*negatives))
    concordant = ((scores>0.) & labels) | ((scores<0.) & ~labels)
    accuracy = float((concordant.sum()+.5*(scores==0).sum())/len(scores)) if len(scores) else None
    return dict(n=int(len(labels)), positives=positives, negatives=negatives,
                auc=auc, sign_accuracy=accuracy)


def choose(scores, eligible, threshold):
    if threshold is None:
        return np.zeros_like(eligible,dtype=bool)
    return eligible & (scores>threshold)


def threshold_from_source(fit_scores, fit_eligible, scores, eligible, ious, quantiles):
    positives = fit_scores[fit_eligible & (fit_scores>0.)]
    levels = [0.] if len(positives)==0 else sorted({0.,*map(float,np.quantile(positives,quantiles))})
    audit, best = [], (0, 0., 0)
    pick = None
    for level in [None,*levels]:
        switched = choose(scores,eligible,level)
        output = np.where(switched,ious[:,1],ious[:,0])
        delta25 = int((output>.25).sum() - (ious[:,0]>.25).sum())
        delta50 = int((output>.50).sum() - (ious[:,0]>.50).sum())
        change = float((output-ious[:,0]).sum())
        count = int(switched.sum())
        allowed = delta25>=0 and change>=-1e-12
        audit.append(dict(threshold=level,switches=count,net25=delta25,net50=delta50,
                          miou_delta_pp=100*change/len(output),eligible_for_selection=allowed))
        key = (delta50,change,-count)
        if allowed and key>best:
            pick,best = level,key
    return pick,audit
