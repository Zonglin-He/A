"""F34: GT-free, mapped observation teachers and genuine episodic students.

This file has no evaluator, source/cohort identifiers, data-outcome readers or
production mutations. Frozen-prefix replay recomputes the entire native suffix.
"""
import copy
import hashlib
import math
import re
import time

import numpy as np
import torch

from vg_tta.closure_replay_v1 import ClosureReplay
from vg_tta.decota_tastvg_episode_v1 import fitted_merge
from vg_tta.shared_state_v1 import norm_of
from vg_tta.time_space_repair_v1 import legal_logp
from vg_tta.metrics import generalized_box_iou_aligned_cxcywh


def tensor_hash(x):
    x = x.detach().contiguous().cpu()
    return hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode() + x.numpy().tobytes()).hexdigest()


def clock_map(x, lo, hi, middle=.6, inverse=False):
    u = (np.asarray(x, dtype=np.float64)-lo)/(hi-lo)
    if inverse:
        y = np.where(u <= middle, .5*u/middle, .5+.5*(u-middle)/(1-middle))
    else:
        y = np.where(u <= .5, 2*middle*u, middle+2*(1-middle)*(u-.5))
    return lo+(hi-lo)*y


def warp_grid(ids, caption, middle=.6):
    # Text-only safety rule. It is not a learned semantic classifier.
    words = re.findall(r'\b(?:fast(?:er|est)?|slow(?:ly|er|est)?|quick(?:ly|er)?|rapid(?:ly)?|speed|seconds?|minutes?|hours?|duration|twice as)\b', caption.lower())
    active = not words and middle != .5
    desired = clock_map(ids, ids[0], ids[-1]+1, middle, True) if active else np.asarray(ids, float)
    actual = np.rint(desired).clip(ids[0], ids[-1]).astype(int)
    actual[0], actual[-1] = ids[0], ids[-1]
    assert np.all(np.diff(actual) >= 0)
    return dict(active=active, skip_reason='explicit_speed_or_duration' if words else None,
                unsafe_tokens=words, middle=middle, nominal_ids=list(ids), actual_ids=actual.tolist(),
                desired_ids=desired.tolist(), quantization_error=(actual-desired).tolist(),
                repeated_positions=int(len(actual)-len(set(actual))), max_actual_gap=int(np.diff(actual).max()),
                endpoints_preserved=True, order_preserved=True)


def _linear_bins(x, grid):
    if x <= grid[0]: return [(0, 1.)]
    if x >= grid[-1]: return [(len(grid)-1, 1.)]
    j = int(np.searchsorted(grid, x, side='left'))
    if x == grid[j]: return [(j, 1.)]
    a = float((x-grid[j-1])/(grid[j]-grid[j-1]))
    return [(j-1, 1-a), (j, a)]


def mass_transport(source_ids, target_ids, lo, hi, middle, inverse):
    """Sparse column-stochastic transport of legal interval atoms.

    Endpoint linear binning has <=4 contributions per source atom. Any illegal
    quantized pair is routed to the nearest legal target atom (L1 endpoints),
    explicitly counted. No dropped-mass renormalization. Identity is exact.
    """
    src = np.asarray(source_ids, float); dst = np.asarray(target_ids, float)
    si, sj = np.triu_indices(len(src), 1); di, dj = np.triu_indices(len(dst), 1)
    starts = clock_map(src[si], lo, hi, middle, inverse)
    ends = clock_map(src[sj]+1, lo, hi, middle, inverse)
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(di, dj))}
    ss, tt, ww, routed = [], [], [], []
    for k, (s, e) in enumerate(zip(starts, ends)):
        for a, wa in _linear_bins(s, dst):
            for b, wb in _linear_bins(e, dst+1):
                w = wa*wb
                if w <= 0: continue
                bad = a >= b
                j = int(np.argmin(np.abs(dst[di]-s)+np.abs(dst[dj]+1-e))) if bad else lookup[a, b]
                ss.append(k); tt.append(j); ww.append(w); routed.append(bad)
    sums = np.bincount(ss, weights=ww, minlength=len(si))
    assert np.max(np.abs(sums-1)) < 1e-12
    return dict(source=np.asarray(ss, np.int64), target=np.asarray(tt, np.int64),
                weight=np.asarray(ww), routed=np.asarray(routed), input_size=len(si), output_size=len(di),
                max_column_error=float(np.max(np.abs(sums-1))),
                source_ids=list(source_ids), target_ids=list(target_ids), inverse=inverse, middle=middle)


def move_mass(p, mapping):
    src = torch.as_tensor(mapping['source'], device=p.device)
    dst = torch.as_tensor(mapping['target'], device=p.device)
    w = torch.as_tensor(mapping['weight'], device=p.device, dtype=p.dtype)
    y = p.new_zeros(mapping['output_size']).scatter_add(0, dst, p[src]*w)
    assert abs(float(y.sum().detach()-p.sum().detach())) < 1e-10
    return y


def build_teacher(logits, records, ids, middle=.6, active=True):
    assert len(logits) == len(records) == 4
    p = [legal_logp(z)[0].exp().detach().clone() for z in logits]
    q, targets, maps, diagnostics = [], [], [], []
    for offset in range(2):
        a, b = records[offset]['frame_ids'], records[offset+2]['frame_ids']
        inv = mass_transport(b, a, ids[0], ids[-1]+1, middle if active else .5, True)
        fwd = mass_transport(a, b, ids[0], ids[-1]+1, middle if active else .5, False)
        mapped = move_mass(p[offset+2], inv)
        target = .5*(p[offset]+mapped)
        q.append(target); targets.append(target); maps.append(dict(inverse=inv, forward=fwd))
        routed = torch.as_tensor(inv['routed'], device=p[0].device)
        src = torch.as_tensor(inv['source'], device=p[0].device)
        w = torch.as_tensor(inv['weight'], device=p[0].device)
        diagnostics.append(dict(mapped_mass=float(mapped.sum()), prior_l1=float((mapped-p[offset]).abs().sum()),
            routed_mass=float((p[offset+2][src]*w*routed).sum()), column_error=inv['max_column_error']))
    targets += [move_mass(q[i], maps[i]['forward']).detach() for i in range(2)]
    return dict(targets=targets, q=q, probabilities=p, maps=maps, diagnostics=diagnostics,
                active=active, middle=middle, GT_online=False)


def posterior_decode(q, records, ids):
    starts, ends = [], []
    for p, r in zip(q, records[:2]):
        ij = torch.triu_indices(len(r['frame_ids']), len(r['frame_ids']), 1)
        k = int(p.argmax())
        starts.append(ids.index(r['frame_ids'][int(ij[0, k])]))
        ends.append(ids.index(r['frame_ids'][int(ij[1, k])]))
    return [min(starts), max(ends)]


def temporal_kl(logits, targets):
    losses = []
    for z, q in zip(logits, targets):
        lp, _ = legal_logp(z)
        q = q.to(lp.device).detach()
        keep = q > 0
        losses.append((q[keep]*(q[keep].log()-lp[keep])).sum())
    return torch.stack(losses).mean()


def image_resample(rgb, scale=.9):
    h, w = rgb.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].float()
    y = torch.nn.functional.interpolate(x, size=(max(2, round(h*scale)), max(2, round(w*scale))),
                                        mode='bilinear', align_corners=False, antialias=True)
    y = torch.nn.functional.interpolate(y, size=(h,w), mode='bilinear', align_corners=False, antialias=True)
    return y.round().clamp(0,255).byte()[0].permute(1,2,0).numpy().copy()


def match_observations(first, second, tau=.25):
    """S2 admits hypotheses; geometry only estimates observation correspondence.

    No temporal association or identity oracle. Unmatched/ambiguous hypotheses
    retain an explicit unknown state and contribute zero weight, not background.
    """
    from scipy.optimize import linear_sum_assignment
    from torchvision.ops import box_convert, box_iou
    a, b = first, second
    if not a['accepted'] or not b['accepted']:
        return dict(anchor=None, reason='one_or_both_S2_rejected', matches=[], ambiguous=True)
    ba = torch.as_tensor(a['boxes']).float(); bb = torch.as_tensor(b['boxes']).float()
    sim = box_iou(box_convert(ba, 'cxcywh', 'xyxy'), box_convert(bb, 'cxcywh', 'xyxy')).numpy()
    rows, cols = linear_sum_assignment(-sim)
    matches = []
    for i, j in zip(rows, cols):
        alternatives = [sim[i,k] for k in range(len(bb)) if k != j]+[sim[k,j] for k in range(len(ba)) if k != i]
        margin = float(sim[i,j]-max(alternatives)) if alternatives else 1.
        unambiguous = bool(sim[i,j] >= .5 and margin >= .05)
        scale = ((ba[i,2:]+bb[j,2:])/2).clamp_min(1e-6).repeat(2)
        u = float(((ba[i]-bb[j]).abs()/scale).mean())
        score = .5*(float(a['target_scores'][i])+float(b['target_scores'][j]))
        matches.append(dict(first_index=int(i), second_index=int(j), iou=float(sim[i,j]),
            assignment_margin=margin, unambiguous=unambiguous, disagreement=u,
            score=score, weight=score*math.exp(-u/tau) if unambiguous else 0.,
            box=((ba[i]+bb[j])/2).tolist()))
    # Select by the same S2 target unary, not GT and not student agreement.
    chosen = max(matches, key=lambda m:m['score'])
    anchor = dict(position=a['position'], frame_id=a['frame_id'], box=chosen['box'],
                  weight=chosen['weight'], score=chosen['score'], disagreement=chosen['disagreement']) if chosen['unambiguous'] else None
    return dict(anchor=anchor, reason='paired' if anchor else 'ambiguous_assignment',
                matches=matches, ambiguous=not chosen['unambiguous'])


def weighted_spatial(boxes, anchors, planned, kappa=None, old_mean=False):
    if not anchors: return boxes.sum()*0
    pos = torch.tensor([a['position'] for a in anchors], device=boxes.device)
    target = torch.tensor([a['box'] for a in anchors], device=boxes.device, dtype=boxes.dtype)
    weight = torch.tensor([a.get('weight',1.) for a in anchors], device=boxes.device, dtype=boxes.dtype)
    pred = boxes[pos]
    d = 5*(pred-target).abs().sum(-1)+2*(1-generalized_box_iou_aligned_cxcywh(pred,target))
    if kappa is not None: d = kappa*torch.log1p(d/kappa)
    return (weight*d).sum()/(len(anchors) if old_mean else planned)


class ObservationReplay(ClosureReplay):
    """Four offset executions, one shared set of trainable parameters."""
    def __init__(self, model, views, n, scope):
        super().__init__(model, views, n, scope, 'fp32', False)
        assert len(views) in (2,4)


def fit_student(it, anchors, teacher, records, ids, *, kind, steps, lrs,
                planned=4, kappa=None, lambda_s=1., gamma=1e-4, old_mean=False):
    """Actual shared optimizer + exact after-state line search; no outcome use."""
    it.restore(it.initial); start=time.perf_counter()
    groups=sorted(set(it.groups.values()))
    opt=torch.optim.Adam([dict(params=[p for n,p in it.named if it.groups[n]==g],
        lr=lrs[g],eps=1e-8 if g=='spatial' else 1e-4) for g in groups],weight_decay=0.)
    with torch.no_grad(): zero=it.values()
    ref=[legal_logp(z)[0].exp().detach() for z in zero['logits']]
    active_s=kind in ('spatial','joint');active_t=kind in ('temporal','joint','coverage')
    empty_s=not any(a.get('weight',1)>0 for a in anchors)
    empty_t=not teacher or not teacher.get('active',True)
    skipped=(not active_s or empty_s) and (not active_t or empty_t) and kind!='coverage'
    def objective(v):
        s=weighted_spatial(v['boxes'],anchors,planned,kappa,old_mean) if active_s else v['boxes'].sum()*0.
        if kind=='coverage':
            from vg_tta.fullspan_tta import fullspan_prior_loss
            t=torch.stack([fullspan_prior_loss(z) for z in v['logits'][:2]]).mean()
        elif active_t and not empty_t: t=temporal_kl(v['logits'],teacher['targets'])
        else: t=v['logits'][0].sum()*0.
        reg=sum(torch.stack([(p-it.initial[n]).square().sum() for n,p in it.named if it.groups[n]==g]).sum()/
                sum(p.numel() for n,p in it.named if it.groups[n]==g) for g in groups)*gamma
        return t+lambda_s*s+reg,dict(temporal=float(t.detach()),spatial=float(s.detach()),regularizer=float(reg.detach()))
    best=math.inf;selected=0;saved=it.state();path=[];failure=None;backwards=0
    for step in range((0 if skipped else steps)+1):
        opt.zero_grad(set_to_none=True);v=it.values();loss,parts=objective(v);lv=float(loss.detach())
        if not math.isfinite(lv) or any(not bool(torch.isfinite(z).all()) for z in [v['boxes']]+v['logits']):
            failure='nonfinite';break
        if lv<best: best=lv;selected=step;saved=it.state()
        st=dict(step=step,loss=lv,parts=parts,boxes=v['boxes'].detach().cpu(),
            logits=[z.detach().cpu() for z in v['logits']],gates=copy.deepcopy(v['gates']),
            indices=list(fitted_merge(v['logits'][:2],records[:2],ids)),
            group_delta={g:norm_of([p-it.initial[n] for n,p in it.named if it.groups[n]==g]) for g in groups},
            posterior_l1=[float((legal_logp(z)[0].exp()-r).abs().sum().detach()) for z,r in zip(v['logits'],ref)])
        path.append(st)
        if step==steps or skipped: break
        loss.backward();backwards+=1
        st['gradient_norms']={g:norm_of([p.grad for n,p in it.named if it.groups[n]==g]) for g in groups}
        unused=[n for n,p in it.named if p.grad is None]
        st['unused_parameters']=unused
        # An ablated loss can legitimately leave its private output unused.
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for n,p in it.named):
            failure='nonfinite_gradient';break
        before=it.state();opt_before=copy.deepcopy(opt.state_dict());opt.step();after=it.state()
        trials=[];accepted=False
        for alpha in (1.,.5,.25,.125):
            it.restore(after if alpha==1 else {n:p+alpha*(after[n]-p) for n,p in before.items()})
            with torch.no_grad(): trial=it.values();newloss,_=objective(trial)
            nv=float(newloss) if bool(torch.isfinite(newloss)) else None
            trials.append(dict(alpha=alpha,loss=nv))
            if nv is not None and nv<lv-1e-9: accepted=True;break
        if not accepted: it.restore(before);opt.load_state_dict(opt_before)
        st.update(accepted=accepted,trials=trials,group_update={g:norm_of([p-before[n] for n,p in it.named if it.groups[n]==g]) for g in groups})
        del loss,v,trial
    assert path,'invalid initialization'
    if failure: selected=0;saved=it.initial
    it.restore(saved)
    with torch.no_grad(): final=it.values()
    assert torch.equal(final['boxes'].cpu(),path[selected]['boxes'])
    assert all(torch.equal(a.cpu(),b) for a,b in zip(final['logits'],path[selected]['logits']))
    result=dict(kind=kind,scope=it.scope,steps=steps,lrs=lrs,lambda_s=lambda_s,gamma=gamma,kappa=kappa,
        planned=planned,old_mean=old_mean,anchors=anchors,parameter_count=sum(p.numel() for n,p in it.named),
        parameter_names={n:p.numel() for n,p in it.named},path=path,best_step=selected,final=path[selected],
        state={n:p.detach().cpu().clone() for n,p in it.named},backwards=backwards,failure=failure,
        skipped='no_effective_supervision' if skipped else None,seconds=time.perf_counter()-start,
        state_delta=norm_of([p-it.initial[n] for n,p in it.named]),GT_online=False,
        student_output_only=True,coverage_coefficient=1 if kind=='coverage' else 0,restore_exact=True)
    it.restore(it.initial)
    assert all(torch.equal(p,it.initial[n]) for n,p in it.named)
    return result
