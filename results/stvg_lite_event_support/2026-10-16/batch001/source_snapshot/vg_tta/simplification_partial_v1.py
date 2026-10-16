"""F36 bounded experimental objectives. No labels, cohort selection or promotion.

Partial/censored observations are hypotheses, not known event labels. Every
gradient below uses the original full-video student. Physical IDs are retained.
"""
import copy
import math
import time

import numpy as np
import torch

from vg_tta.parametric_observation_v1 import weighted_spatial, ObservationReplay
from vg_tta.shared_state_v1 import norm_of
from vg_tta.time_space_repair_v1 import legal_logp
from vg_tta.decota_tastvg_episode_v1 import fitted_merge


def crop_windows(ids, interval, bounds):
    """Snap by selecting original positions; never resample or pad a crop.

    Bounds of the actual selected support are [first ID,last ID+1). This is
    also the physical support of native postprocessing. Full inputs and exact
    duplicates are removed before any teacher inference or loss weighting.
    """
    ids = np.asarray(ids, dtype=int)
    lo, hi = bounds
    s, e = ids[interval[0]], ids[interval[1]]+1
    requested = [(lo, max(lo+.75*(hi-lo), e)),
                 (min(lo+.25*(hi-lo), s), hi)]
    out, seen, ignored = [], set(), []
    for name, (left, right) in zip(('left', 'right'), requested):
        pos = np.flatnonzero((ids >= left) & (ids < right)).tolist()
        reason = 'full_input' if len(pos) == len(ids) else 'duplicate' if tuple(pos) in seen else 'too_short' if len(pos) < 4 else None
        if reason:
            ignored.append(dict(name=name, reason=reason, requested=[left, right], positions=pos))
            continue
        seen.add(tuple(pos))
        # Delta is the median effective per-offset interval (not a tuned GT tolerance).
        diffs = np.concatenate([np.diff(ids[pos][j::2]) for j in (0, 1)])
        out.append(dict(name=name, requested=[float(left), float(right)], positions=pos,
                        frame_ids=ids[pos].tolist(), bounds=[int(ids[pos[0]]), int(ids[pos[-1]]+1)],
                        delta=float(np.median(diffs)), merged_delta=float(np.median(np.diff(ids[pos])))))
    return dict(windows=out, ignored=ignored, planned=2)


def observation(window, interval, kind='partial'):
    a, b = map(float, interval)
    left, right = window['bounds']; d = window['delta']
    assert left <= a < b <= right
    return dict(bounds=[left, right], interval=[a, b], delta=d,
                left_censored=kind != 'point' and a-left <= d,
                right_censored=kind != 'point' and right-b <= d,
                kind=kind, name=window['name'])


def compatible(ids, obs):
    ids = np.asarray(ids, dtype=float)
    ii, jj = np.triu_indices(len(ids), 1)
    s, e = ids[ii], ids[jj]+1
    left, right = obs['bounds']; a, b = obs['interval']; d = obs['delta']
    start = s <= left+d if obs['left_censored'] else np.abs(s-a) <= d
    end = e >= right-d if obs['right_censored'] else np.abs(e-b) <= d
    mask = start & end
    if not mask.any():
        raise ValueError(('empty_compatible_set_construction_error', ids.tolist(), obs))
    return torch.from_numpy(mask)


def make_supervision(records, observations):
    assert len(records) == 2 and len(observations) <= 2
    signatures = [(tuple(o['bounds']), tuple(o['interval']), o['left_censored'], o['right_censored']) for o in observations]
    assert len(set(signatures)) == len(signatures), 'duplicate observation reached objective'
    masks = [[compatible(r['frame_ids'], o) for o in observations] for r in records]
    return dict(observations=copy.deepcopy(observations), masks=masks, planned=2, epsilon=.05,
                active=any(not bool(m.all()) for mm in masks for m in mm), GT_online=False)


def partial_objective(logits, reference_logp, teacher, beta):
    data = logits[0].sum()*0.
    prior = data
    masses = []
    for z, ref, masks in zip(logits, reference_logp, teacher['masks']):
        lp, _ = legal_logp(z); p = lp.exp(); ref = ref.to(lp.device)
        prior = prior+(p*(lp-ref)).sum()/len(logits)
        mm = []
        for mask in masks:
            mask = mask.to(lp.device)
            mass = p[mask].sum()
            if bool(mask.all()):
                # Exact uninformative observation, not roundoff self-training.
                mass = mass*0+1
            data = data-torch.log(.05+.95*mass)/(2*len(logits))
            mm.append(float(mass.detach()))
        masses.append(mm)
    return data+beta*prior, dict(data=float(data.detach()), prior_KL=float(prior.detach()),
                                beta=beta, masses=masses)


def convex_output(reference_logp, teacher, beta):
    """Same objective on the free legal simplex, solved in <=2 dual variables.

    p_i proportional p0_i exp(sum_v lambda_v A_vi). The dual is strictly convex.
    Stationarity and primal/dual gap are reported; this is not interval argmax.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    outputs, audit = [], []
    tick = time.perf_counter()
    for ref, masks in zip(reference_logp, teacher['masks']):
        lp0 = ref.detach().cpu().double().numpy()
        A = np.array([m.numpy().astype(float) for m in masks if not bool(m.all())])
        constants = 0
        if not len(A):
            outputs.append(torch.from_numpy(np.exp(lp0)))
            audit.append(dict(converged=True, gap=0., gradient_inf=0., informative=0))
            continue
        a, c, eps = .5, .95, .05
        def objective(lam):
            z = lp0+lam@A; lz = logsumexp(z); p = np.exp(z-lz)
            f = beta*lz-a*np.log(lam).sum()+beta*eps/c*lam.sum()
            g = beta*(A@p)-a/lam+beta*eps/c
            return f, g
        opt = minimize(objective, np.full(len(A), a/beta), jac=True, method='L-BFGS-B',
                       bounds=[(1e-12, a*c/(beta*eps))]*len(A),
                       options=dict(maxiter=2000, ftol=1e-15, gtol=1e-10, maxls=50))
        lam = opt.x; z = lp0+lam@A; lz = logsumexp(z); lp = z-lz; p = np.exp(lp)
        m = A@p
        primal = -a*np.log(eps+c*m).sum()+beta*np.sum(p*(lp-lp0))
        # Conjugate at u=beta*lambda/c. g(m)>=-u*(eps+c*m)+a+a log(u/a).
        dual = -beta*lz + np.sum(-beta*lam*eps/c+a+a*np.log(beta*lam/(c*a)))
        gap = float(primal-dual)
        gradient = float(np.abs(objective(lam)[1]).max())
        assert gap >= -1e-9 and gap < 1e-7, ('simplex_not_converged', gap, gradient, opt.message)
        outputs.append(torch.from_numpy(p))
        audit.append(dict(converged=gap < 1e-7, solver_success=bool(opt.success), gap=gap,
                          gradient_inf=gradient, informative=len(A), iterations=int(opt.nit),
                          primal=float(primal), dual=float(dual), lambda_values=lam.tolist(),
                          masses=m.tolist(), sum_error=float(abs(p.sum()-1))))
    return dict(q=outputs, audit=audit, seconds=time.perf_counter()-tick, parameter_update=False)


class FullInputHeadReplay:
    """Exact head-only factorization: frozen full-input features, updated real head.

    Capture at the actual FP32 temporal head input, not the earlier shared H.
    Compare every baseline tensor and reinsert each final state into the model.
    """
    def __init__(self, replay):
        self.replay = replay
        self.head = replay.head
        self.named, self.groups, self.scope = replay.named, replay.groups, replay.scope
        self.initial = replay.initial
        self.inputs = []
        h = self.head.register_forward_pre_hook(lambda m, args: self.inputs.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                self.zero = replay.values()
        finally:
            h.remove()
        assert len(self.inputs) == 2
        with torch.no_grad():
            v = self.values()
        assert all(torch.equal(a, b) for a, b in zip(v['logits'], self.zero['logits']))
    def state(self):
        return self.replay.state()
    def restore(self, state):
        return self.replay.restore(state)
    def values(self):
        return {**self.zero, 'logits': [self.head(h)[-1] for h in self.inputs]}


def fit(it, records, ids, *, anchors=None, planned=4, kappa=None, gamma=0.,
        teacher=None, beta=1., lr=.01, steps=10, charge=None):
    """Actual Adam proposal, after-state backtracking, rollback and best state.

    No predictive utility/GT appears in selection. Gamma uses parent denominator.
    No extra VJP: data gradient is total gradient minus analytic regularizer.
    """
    it.restore(it.initial)
    if charge: charge('fits', 1)
    start = time.perf_counter(); named = it.named; params = [p for n, p in named]
    temporal = teacher is not None
    eps = 1e-4 if temporal else 1e-8
    opt = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=0.)
    with torch.no_grad(): zero = it.values()
    refs = [legal_logp(z)[0].detach().clone() for z in zero['logits']]
    skipped = not teacher['active'] if temporal else not anchors
    def objective(v):
        if temporal:
            return partial_objective(v['logits'], refs, teacher, beta)
        data = weighted_spatial(v['boxes'], anchors, planned, kappa)
        reg = sum((p-it.initial[n]).square().sum() for n, p in named)/1792*gamma
        return data+reg, dict(data=float(data.detach()), regularizer=float(reg.detach()))
    best, selected, saved = math.inf, 0, it.state()
    path, failure, backwards = [], None, 0
    forwards, forward_s, backtrack_s = 0, 0., 0.
    def values():
        nonlocal forwards, forward_s
        t = time.perf_counter(); v = it.values()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        forwards += 1; forward_s += time.perf_counter()-t
        return v
    for step in range((0 if skipped else steps)+1):
        opt.zero_grad(set_to_none=True); v = values(); loss, parts = objective(v); lv = float(loss.detach())
        if not math.isfinite(lv): failure = 'nonfinite'; break
        if lv < best: best, selected, saved = lv, step, it.state()
        st = dict(step=step, loss=lv, parts=parts, boxes=v['boxes'].detach().cpu(),
                  logits=[z.detach().cpu() for z in v['logits']],
                  indices=list(fitted_merge(v['logits'], records, ids)),
                  state_delta=norm_of([p-it.initial[n] for n, p in named]))
        if not temporal:
            bb = v['boxes'].detach().clamp(1e-7, 1-1e-7)
            st['regression_logit_range'] = [float(torch.logit(bb).min()), float(torch.logit(bb).max())]
        path.append(st)
        if skipped or step == steps: break
        if charge: charge('backward', 1)
        loss.backward(); backwards += 1
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
            failure = 'nonfinite_gradient'; break
        st['gradient_norm'] = norm_of([p.grad for p in params])
        st['data_gradient_norm'] = norm_of([(torch.zeros_like(p) if p.grad is None else p.grad)-
                                           (0 if temporal else 2*gamma/1792*(p-it.initial[n])) for n, p in named])
        st['unused_parameters'] = [n for n, p in named if p.grad is None]
        before, opt_before = it.state(), copy.deepcopy(opt.state_dict())
        opt.step(); after = it.state(); st['proposal_norm'] = norm_of([after[n]-before[n] for n, p in named])
        trials, accepted = [], False; tick = time.perf_counter()
        for alpha in (1., .5, .25, .125):
            it.restore(after if alpha == 1 else {n: p+alpha*(after[n]-p) for n, p in before.items()})
            with torch.no_grad(): trial = values(); newloss, _ = objective(trial)
            nv = float(newloss); trials.append(dict(alpha=alpha, loss=nv if math.isfinite(nv) else None))
            if math.isfinite(nv) and nv < lv-1e-9: accepted = True; break
        if not accepted:
            it.restore(before); opt.load_state_dict(opt_before)
        backtrack_s += time.perf_counter()-tick
        st.update(accepted=accepted, trials=trials, update_norm=norm_of([p-before[n] for n, p in named]))
        del loss, v, trial
    assert path
    if failure: selected, saved = 0, it.initial
    it.restore(saved)
    with torch.no_grad(): final = values()
    assert torch.equal(final['boxes'].cpu(), path[selected]['boxes'])
    assert all(torch.equal(a.cpu(), b) for a, b in zip(final['logits'], path[selected]['logits']))
    result = dict(kind='temporal' if temporal else 'spatial', steps=steps, lr=lr, eps=eps,
        gamma=gamma, kappa=kappa, planned=planned, beta=beta if temporal else None,
        anchors=copy.deepcopy(anchors), path=path, best_step=selected, final=path[selected],
        state={n: p.detach().cpu().clone() for n, p in named},
        parameter_count=sum(p.numel() for p in params), parameter_names={n:p.numel() for n,p in named},
        failure=failure, skipped=bool(skipped), backwards=backwards, suffix_forwards=forwards,
        suffix_seconds=forward_s, backtrack_seconds=backtrack_s, seconds=time.perf_counter()-start,
        state_delta=norm_of([p-it.initial[n] for n,p in named]), GT_online=False,
        restore_exact=True, student_output_only=True, full_input_gradients_only=True)
    it.restore(it.initial)
    assert all(torch.equal(p, it.initial[n]) for n,p in named)
    return result
