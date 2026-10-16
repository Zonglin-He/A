"""Read-only trajectory diagnostics; no change to the deployed method.

All boxes are normalized cxcywh. Ground truth is used only in diagnostics.
Motion is measured in original pixels and physical seconds, not sample index.
"""
import numpy as np


def overlap(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo = np.maximum(a[..., :2] - a[..., 2:] / 2, b[..., :2] - b[..., 2:] / 2)
    hi = np.minimum(a[..., :2] + a[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2)
    inter = np.maximum(hi-lo, 0).prod(-1)
    union = a[..., 2:].prod(-1) + b[..., 2:].prod(-1) - inter
    return inter / np.maximum(union, 1e-12)


def smooth(boxes, times, present, *, window=None, seconds=None, median=False):
    """GT-free symmetric, offline smoothers; never fill missing output boxes."""
    boxes, times, present = np.asarray(boxes, float), np.asarray(times, float), np.asarray(present, bool)
    out = boxes.copy()
    for i in np.flatnonzero(present):
        if window is not None:
            take = np.arange(max(0, i-window//2), min(len(boxes), i+window//2+1))
            take = take[present[take]]
            out[i] = np.median(boxes[take], axis=0) if median else boxes[take].mean(0)
        else:
            take = np.flatnonzero(present & (np.abs(times-times[i]) <= seconds))
            weights = np.maximum(1-np.abs(times[take]-times[i])/seconds, 0)
            out[i] = np.average(boxes[take], axis=0, weights=weights)
    return out


def iou_change_accounting(p0, p1, g0, g1):
    """Symmetric two-order accounting, not causal attribution."""
    a, b, c, d = overlap(p0,g0), overlap(p1,g0), overlap(p0,g1), overlap(p1,g1)
    pred = .5*((b-a)+(d-c)); gt = .5*((c-a)+(d-b))
    assert np.allclose(pred+gt, d-a, atol=1e-12)
    return pred, gt


def curvature(c, times, eligible):
    """Deviation from local constant velocity at irregular time positions."""
    take = np.flatnonzero(eligible[:-2] & eligible[1:-1] & eligible[2:]) + 1
    if not len(take): return None
    w = (times[take]-times[take-1])/(times[take+1]-times[take-1])
    linear = (1-w[:,None])*c[take-1] + w[:,None]*c[take+1]
    return float(np.linalg.norm(c[take]-linear, axis=1).mean())


def residual_decomposition(pred, gt, times, valid, shape):
    """Orthogonal SSE partition: constant, time trend, parity, remainder."""
    take = np.flatnonzero(valid); scale = np.asarray(shape, float)
    y = (pred[take,:2]-gt[take,:2])*scale
    denom = float((y*y).sum())
    if len(take)<4 or denom<1e-16: return None
    t = times[take]-times[take].mean(); t /= max(float(np.std(t)),1e-12)
    parity = (take % 2)*2.-1.
    remainder = y.copy(); fractions = {}; fits = []
    for name, column in [('constant', np.ones(len(take))), ('linear_drift',t), ('offset_parity',parity)]:
        v = column.copy()
        for q in fits: v -= q*(q@v)
        norm = np.linalg.norm(v)
        if norm<1e-10: fractions[name]=0.; continue
        v /= norm; component = v[:,None]*(v@remainder)[None,:]
        fractions[name] = float((component*component).sum()/denom)
        remainder -= component; fits.append(v)
    fractions['remainder'] = float((remainder*remainder).sum()/denom)
    assert abs(sum(fractions.values())-1)<1e-8
    bias = y.mean(0)/scale
    corrected = pred.copy(); corrected[:,:2] -= bias
    return dict(fractions=fractions,constant_error_px=float(np.linalg.norm(y.mean(0))),
                oracle_constant_offset_sIoU=float(overlap(corrected[take],gt[take]).mean()))


def trajectory(boxes, truth, valid, frame_ids, fps, shape, present=None):
    b, g = np.asarray(boxes,float), np.asarray(truth,float)
    valid = np.asarray(valid,bool); ids=np.asarray(frame_ids,float); times=ids/fps
    present = np.ones(len(b),bool) if present is None else np.asarray(present,bool)
    assert np.all(np.diff(ids)>0) and fps>0 and b.shape==g.shape
    b = b.copy(); b[~present]=0
    q = overlap(b,g); eligible=valid & present
    adjacent = valid[:-1] & valid[1:] & present[:-1] & present[1:]
    jj=np.flatnonzero(adjacent); dt=np.diff(times)[jj]; shape=np.asarray(shape,float)
    pc=b[:,:2]*shape; gc=g[:,:2]*shape
    diag=np.linalg.norm(g[:,2:]*shape,axis=1)
    mean=lambda z: float(np.mean(z)) if len(z) else None
    out=dict(sIoU=float(q[valid].mean()),GT_frames=int(valid.sum()),present_GT_frames=int(eligible.sum()),
             IoU_std=float(q[valid].std()),IoU_range=float(np.ptp(q[valid])),
             adjacent_pairs=len(jj),sampling_gap_frames=float(np.median(np.diff(ids))),
             sampling_gap_seconds=float(np.median(np.diff(times))),
             adjacent_original_frame_fraction=float(np.mean(np.diff(ids)==1)))
    if not len(jj): return out, []
    dp=np.linalg.norm(pc[jj+1]-pc[jj],axis=1); dg=np.linalg.norm(gc[jj+1]-gc[jj],axis=1)
    dr=np.linalg.norm((pc-gc)[jj+1]-(pc-gc)[jj],axis=1)
    norm=np.maximum((diag[jj]+diag[jj+1])/2,1e-6)
    pq=overlap(b[jj],b[jj+1]); gq=overlap(g[jj],g[jj+1]); dq=np.abs(q[jj+1]-q[jj])
    pa,ga=iou_change_accounting(b[jj],b[jj+1],g[jj],g[jj+1])
    pred_curv=curvature(pc,times,eligible); gt_curv=curvature(gc,times,eligible)
    out.update(pred_step_px=mean(dp),GT_step_px=mean(dg),error_step_px=mean(dr),
        pred_speed_px_s=mean(dp/dt),GT_speed_px_s=mean(dg/dt),pred_step_target_diag=mean(dp/norm),
        GT_step_target_diag=mean(dg/norm),pred_adjacent_IoU=mean(pq),GT_adjacent_IoU=mean(gq),
        abs_IoU_step=mean(dq),large_IoU_step_fraction=mean(dq>.3),
        GT_stable_pair_fraction=mean(gq>=.9),
        large_IoU_step_given_GT_stable=mean(dq[gq>=.9]>.3),
        pred_jump_given_GT_stable=mean(pq[gq>=.9]<.5),
        pred_curvature_px=pred_curv,GT_curvature_px=gt_curv,
        abs_pred_accounting=mean(np.abs(pa)),abs_GT_accounting=mean(np.abs(ga)),
        normalized_center_error=mean(np.linalg.norm(pc[eligible]-gc[eligible],axis=1)/np.maximum(diag[eligible],1e-6)))
    dec=residual_decomposition(b,g,times,eligible,shape)
    if dec:
        out.update({'center_SSE_'+k:v for k,v in dec['fractions'].items()})
        out['oracle_constant_offset_delta_sIoU']=dec['oracle_constant_offset_sIoU']-float(q[eligible].mean())
    pairs=[dict(i=int(j),frame0=int(ids[j]),frame1=int(ids[j+1]),seconds=float(dt[k]),
                pred_step_px=float(dp[k]),GT_step_px=float(dg[k]),error_step_px=float(dr[k]),
                GT_diag_px=float(norm[k]),pred_adjacent_IoU=float(pq[k]),GT_adjacent_IoU=float(gq[k]),
                IoU0=float(q[j]),IoU1=float(q[j+1]),abs_IoU_step=float(dq[k]),
                pred_accounting=float(pa[k]),GT_accounting=float(ga[k])) for k,j in enumerate(jj)]
    return out,pairs
