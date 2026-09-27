"""F35 bounded, GT-free single-factor experiments; parent Spatial10 is immutable.

This is an instrumented experimental runner, not a promoted production method.
It preserves original detach, FP32 suffix and actual-after Adam semantics.
"""
import copy
import math
import time

import numpy as np
import torch

from vg_tta.parametric_observation_v1 import weighted_spatial, temporal_kl, tensor_hash
from vg_tta.shared_state_v1 import norm_of
from vg_tta.decota_tastvg_episode_v1 import fitted_merge
from vg_tta.metrics import generalized_box_iou_aligned_cxcywh


def scope_mask(it, scope):
    if scope == 'both':
        return
    keep = []
    for name, p in it.named:
        selected = (name == 'spatial.query_residual') == (scope == 'query')
        p.requires_grad_(selected)
        if selected:
            keep.append((name, p))
    assert scope in ('query', 'ln')
    it.named = keep
    it.groups = {n: 'spatial' for n, p in keep}
    it.initial = it.state()
    it.scope = scope


def anchor_variant(anchors, variant, seed=None):
    aa = copy.deepcopy(anchors)
    if not aa:
        return aa
    if variant == 'confidence':
        for a in aa:
            a['weight'] = a['score']
    elif variant == 'uniform':
        w = sum(a['weight'] for a in aa)/len(aa)
        for a in aa:
            a['weight'] = w
    elif variant == 'shuffle':
        weights = np.array([a['weight'] for a in aa])
        np.random.default_rng(seed).shuffle(weights)
        for a, w in zip(aa, weights):
            a['weight'] = float(w)
    elif variant != 'original':
        raise ValueError(variant)
    if variant in ('uniform', 'shuffle'):
        assert abs(sum(a['weight'] for a in aa)-sum(a['weight'] for a in anchors)) < 1e-12
    return aa


def distances(boxes, anchors):
    pos = torch.tensor([a['position'] for a in anchors], device=boxes.device)
    target = boxes.new_tensor([a['box'] for a in anchors])
    pred = boxes[pos]
    return 5*(pred-target).abs().sum(-1)+2*(1-generalized_box_iou_aligned_cxcywh(pred, target))


def vector(tensors, parameters):
    return torch.cat([(torch.zeros_like(p) if x is None else x).detach().reshape(-1)
                      for x, p in zip(tensors, parameters)])


def cosine(a, b):
    denom = float(a.double().norm()*b.double().norm())
    return float((a.double()*b.double()).sum()/denom) if denom else None


def moment_summary(opt, named):
    out = {}
    for n, p in named:
        st = opt.state.get(p, {})
        out[n] = {k: (float(v) if v.numel() == 1 else
                       dict(norm=norm_of([v]), sha256=tensor_hash(v)))
                  for k, v in st.items() if torch.is_tensor(v)}
    return out


def fit(it, anchors, records, ids, *, lr, steps=10, kappa=None, gamma=1e-4,
        gradients=False, frozen_influence=False, kind='spatial', teacher=None,
        head_lr=.001, independent_accept=False, charge=None):
    """One actual path; count both backward and diagnostic VJP traversals.

    Spatial regularization denominator remains 1792 even under Q/LN masks.
    T2 restores group-specific Adam states on rejection and selects own best.
    No teacher is selected or altered by an evaluation outcome.
    """
    it.restore(it.initial)
    if charge:
        charge('fits', 1)
    start = time.perf_counter()
    groups = sorted(set(it.groups.values()))
    named = it.named
    params = [p for n, p in named]
    group_params = {g: [(n, p) for n, p in named if it.groups[n] == g] for g in groups}
    opt = torch.optim.Adam([dict(params=[p for n, p in group_params[g]],
                          lr=lr if g == 'spatial' else head_lr,
                          eps=1e-8 if g == 'spatial' else 1e-4) for g in groups], weight_decay=0.)
    calls = 0
    forward_seconds = 0.
    backtrack_seconds = 0.
    def values():
        nonlocal calls, forward_seconds
        t = time.perf_counter()
        z = it.values()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        calls += 1
        forward_seconds += time.perf_counter()-t
        return z
    with torch.no_grad():
        zero = values()
        influence = 1/(1+distances(zero['boxes'], anchors)/2) if frozen_influence and anchors else None
    active_s = kind in ('spatial', 'joint')
    active_t = kind in ('temporal', 'joint')
    empty_s = not any(a.get('weight', 1) > 0 for a in anchors)
    empty_t = not teacher or not teacher.get('active', True)
    skipped = (not active_s or empty_s) and (not active_t or empty_t)
    denoms = {g: 1792 if g == 'spatial' else sum(p.numel() for n, p in group_params[g]) for g in groups}
    def objective(v):
        if active_s and influence is not None:
            s = (v['boxes'].new_tensor([a['weight'] for a in anchors])*influence*
                 distances(v['boxes'], anchors)).sum()/4
        else:
            s = weighted_spatial(v['boxes'], anchors, 4, kappa, False) if active_s else v['boxes'].sum()*0.
        t = temporal_kl(v['logits'], teacher['targets']) if active_t and not empty_t else v['logits'][0].sum()*0.
        # Same arithmetic order as the parent for exact C0 reproduction.
        rg = {g: torch.stack([(p-it.initial[n]).square().sum() for n, p in group_params[g]]).sum()/denoms[g]
              for g in groups}
        reg = sum(rg.values())*gamma
        loss = t+s+reg
        gp = {g: (s if g == 'spatial' else t)+rg[g]*gamma for g in groups}
        parts = dict(temporal=float(t.detach()), spatial=float(s.detach()), regularizer=float(reg.detach()),
                     group_reg={g: float(rg[g].detach()*gamma) for g in groups},
                     group_loss={g: float(z.detach()) for g, z in gp.items()})
        return loss, s+t, reg, gp, parts
    best = math.inf
    selected = 0
    saved = it.state()
    best_group = {g: math.inf for g in groups}
    selected_group = {g: 0 for g in groups}
    saved_group = it.state()
    path = []
    failure = None
    backwards = 0
    extra_vjp = 0
    for step in range((0 if skipped else steps)+1):
        opt.zero_grad(set_to_none=True)
        v = values()
        loss, data, reg, gl, parts = objective(v)
        lv = float(loss.detach())
        if not math.isfinite(lv) or any(not bool(torch.isfinite(z).all()) for z in [v['boxes']]+v['logits']):
            failure = 'nonfinite'
            break
        if lv < best:
            best, selected, saved = lv, step, it.state()
        for g in groups:
            if parts['group_loss'][g] < best_group[g]:
                best_group[g], selected_group[g] = parts['group_loss'][g], step
                for n, p in group_params[g]:
                    saved_group[n] = p.detach().clone()
        st = dict(step=step, loss=lv, parts=parts, boxes=v['boxes'].detach().cpu(),
                  logits=[z.detach().cpu() for z in v['logits']], gates=copy.deepcopy(v['gates']),
                  indices=list(fitted_merge(v['logits'][:2], records[:2], ids)),
                  group_delta={g: norm_of([p-it.initial[n] for n, p in group_params[g]]) for g in groups})
        path.append(st)
        if step == steps or skipped:
            break
        if gradients:
            if charge:
                charge('backward', 1)
            gd = torch.autograd.grad(data, params, allow_unused=True, retain_graph=True)
            extra_vjp += 1
            dv = vector(gd, params)
            rv = vector([2*gamma/denoms[it.groups[n]]*(p-it.initial[n]) for n, p in named], params)
        if charge:
            charge('backward', 1)
        loss.backward()
        backwards += 1
        gv = vector([p.grad for p in params], params)
        st['gradient_norms'] = {g: norm_of([p.grad for n, p in group_params[g]]) for g in groups}
        st['unused_parameters'] = [n for n, p in named if p.grad is None]
        if not bool(torch.isfinite(gv).all()):
            failure = 'nonfinite_gradient'
            break
        if gradients:
            st['gradient_decomposition'] = dict(data=dv.cpu(), regularizer=rv.cpu(),
                data_norm=norm_of([dv]), reg_norm=norm_of([rv]), cosine=cosine(dv, rv),
                sum_max_error=float((gv-dv-rv).abs().max()))
        before = it.state()
        opt_before = copy.deepcopy(opt.state_dict())
        st['moments_before'] = moment_summary(opt, named) if kind != 'spatial' else None
        opt.step()
        after = it.state()
        proposal = vector([after[n]-before[n] for n, p in named], params)
        st['proposal'] = {n: (after[n]-before[n]).cpu() for n, p in named} if kind != 'spatial' else None
        st['proposal_norm'] = norm_of([proposal])
        trials = []
        accepted = False
        tick = time.perf_counter()
        if independent_accept:
            # Trial each group against its own loss, never jointly accept it
            # because another task compensates. Moment restoration is per group.
            accepted_groups = {}
            chosen_alphas = {}
            chosen = copy.deepcopy(before)
            for g in groups:
                accepted_groups[g] = False
                for alpha in (1., .5, .25, .125):
                    test = copy.deepcopy(before)
                    for n, p in group_params[g]:
                        test[n] = after[n] if alpha == 1 else before[n]+alpha*(after[n]-before[n])
                    it.restore(test)
                    with torch.no_grad():
                        trial = values()
                        _, _, _, _, tp = objective(trial)
                    nv = tp['group_loss'][g]
                    trials.append(dict(group=g, alpha=alpha, parts=tp, loss=nv))
                    if math.isfinite(nv) and nv < parts['group_loss'][g]-1e-9:
                        accepted_groups[g] = True
                        chosen_alphas[g] = alpha
                        for n, p in group_params[g]:
                            chosen[n] = test[n]
                        break
                if not accepted_groups[g]:
                    # Restore exactly the pre-proposal Adam state for this group.
                    gi = groups.index(g)
                    old_ids = opt_before['param_groups'][gi]['params']
                    for (_, p), pid in zip(group_params[g], old_ids):
                        if pid in opt_before['state']:
                            opt.state[p] = copy.deepcopy(opt_before['state'][pid])
                        else:
                            opt.state.pop(p, None)
            it.restore(chosen)
            accepted = any(accepted_groups.values())
            st.update(accepted_groups=accepted_groups, chosen_alphas=chosen_alphas)
        else:
            for alpha in (1., .5, .25, .125):
                it.restore(after if alpha == 1 else {n: p+alpha*(after[n]-p) for n, p in before.items()})
                with torch.no_grad():
                    trial = values()
                    nl, _, _, _, tp = objective(trial)
                nv = float(nl) if bool(torch.isfinite(nl)) else None
                trials.append(dict(alpha=alpha, loss=nv, parts=tp))
                if nv is not None and nv < lv-1e-9:
                    accepted = True
                    break
            if not accepted:
                it.restore(before)
                opt.load_state_dict(opt_before)
        backtrack_seconds += time.perf_counter()-tick
        actual = vector([p-before[n] for n, p in named], params)
        st.update(accepted=accepted, trials=trials,
                  group_update={g: norm_of([p-before[n] for n, p in group_params[g]]) for g in groups},
                  actual_displacement=actual.cpu() if gradients else None,
                  moments_after=moment_summary(opt, named) if kind != 'spatial' else None)
        if not accepted:
            assert all(torch.equal(p, before[n]) for n, p in named)
            # State hashes as well as numerical parameters are checked below in tests.
        del loss, data, reg, gl, v, trial
    assert path, 'invalid initialization'
    if independent_accept:
        saved = saved_group
    if failure:
        selected, saved = 0, it.initial
    it.restore(saved)
    with torch.no_grad():
        final = values()
    if not independent_accept:
        assert torch.equal(final['boxes'].cpu(), path[selected]['boxes'])
        assert all(torch.equal(z.cpu(), q) for z, q in zip(final['logits'], path[selected]['logits']))
    else:
        # These assertions are valid only after explicit zero-cross-gradient audit.
        if 'spatial' in groups:
            assert torch.equal(final['boxes'].cpu(), path[selected_group['spatial']]['boxes'])
        if 'head' in groups:
            assert all(torch.equal(z.cpu(), q) for z, q in zip(final['logits'], path[selected_group['head']]['logits']))
    outfinal = dict(boxes=final['boxes'].detach().cpu(), logits=[z.detach().cpu() for z in final['logits']],
                    indices=list(fitted_merge(final['logits'][:2], records[:2], ids)))
    result = dict(scope=it.scope, kind=kind, steps=steps, lr=lr, head_lr=head_lr, eps_spatial=1e-8,
        eps_head=1e-4, gamma=gamma, spatial_reg_denominator=1792, planned_denominator=4,
        kappa=kappa, frozen_influence=influence.cpu() if influence is not None else None,
        anchors=anchors, parameter_count=sum(p.numel() for p in params),
        parameter_names={n: p.numel() for n, p in named}, path=path, best_step=selected,
        best_group_step=selected_group, independent_accept=independent_accept, final=outfinal,
        state={n: p.detach().cpu().clone() for n, p in named}, backwards=backwards, extra_vjp=extra_vjp,
        suffix_forwards=calls, suffix_seconds=forward_seconds, backtrack_seconds=backtrack_seconds,
        failure=failure, skipped='no_effective_supervision' if skipped else None,
        seconds=time.perf_counter()-start, state_delta=norm_of([p-it.initial[n] for n, p in named]),
        GT_online=False, student_output_only=True, restore_exact=True)
    it.restore(it.initial)
    assert all(torch.equal(p, it.initial[n]) for n, p in named)
    return result


def cross_gradients(it, anchors, teacher, kappa, charge=None):
    it.restore(it.initial)
    v = it.values()
    ls = weighted_spatial(v['boxes'], anchors, 4, kappa)
    lt = temporal_kl(v['logits'], teacher['targets'])
    pp = [p for n, p in it.named]
    if charge:
        charge('backward', 2)
    gs = torch.autograd.grad(ls, pp, allow_unused=True, retain_graph=True)
    gt = torch.autograd.grad(lt, pp, allow_unused=True)
    result = dict(spatial_to_head=norm_of([z for (n, p), z in zip(it.named, gs) if it.groups[n] == 'head']),
                  temporal_to_spatial=norm_of([z for (n, p), z in zip(it.named, gt) if it.groups[n] == 'spatial']),
                  spatial_to_spatial=norm_of([z for (n, p), z in zip(it.named, gs) if it.groups[n] == 'spatial']),
                  temporal_to_head=norm_of([z for (n, p), z in zip(it.named, gt) if it.groups[n] == 'head']))
    it.restore(it.initial)
    return result
