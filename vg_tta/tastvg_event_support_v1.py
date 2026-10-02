"""H-lite: detached native temporal consensus conditions reward and geometry.

This is not a calibrated posterior or a reliability score. Uniform candidate
weights stay uniform; no entropy-based loss scaling or new temporal parameters.
"""
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached, state_hash
from vg_tta.tastvg_selected_rollout_v1 import (
    SingleStep as OriginalSingleStep, central_with_candidates, fast_rerank,
    target_loss, project_arrival, apply_direction, functional_movement,
    rollout_states, norm,
)
from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks, update_scale
from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
from vg_tta.box_stability_diagnostics_v1 import overlap


def event_weights(support, frame_ids):
    """Equal votes over actual deduplicated native candidates, half-open frames."""
    assert 0 < len(support) <= 8 and len(frame_ids) > 1
    ids = np.asarray(frame_ids)
    assert np.all(np.diff(ids) > 0)
    spans = np.asarray([c['physical_interval'] for c in support])
    assert np.all(spans[:, 1] > spans[:, 0])
    votes = (ids[None, :] >= spans[:, :1]) & (ids[None, :] < spans[:, 1:])
    w = votes.astype(np.float64).mean(0)
    assert w.sum() > 0 and np.isfinite(w).all()
    return w


def event_rewards(candidate_boxes, expert_boxes, valid, weights):
    """Average only real valid cached references, with native consensus weight."""
    valid = np.asarray(valid, bool)
    w = np.asarray(weights, np.float64)
    assert w.shape == valid.shape and np.isfinite(w).all() and (w >= 0).all()
    mass = float(w[valid].sum())
    if mass == 0:
        return None
    return np.array([np.dot(overlap(np.asarray(b)[valid],
        np.asarray(expert_boxes)[valid]), w[valid])/mass for b in candidate_boxes])


def event_geometry(central, candidates, l1_coef, giou_coef, weights):
    """Original native frame loss, reduced with detached temporal weights."""
    q = candidates.detach(); p = central.unsqueeze(0)
    w = torch.as_tensor(weights, device=central.device, dtype=central.dtype).detach()
    assert w.shape == central.shape[:-1] and torch.isfinite(w).all() and w.sum() > 0
    a, b = p[..., :2]-p[..., 2:]/2, p[..., :2]+p[..., 2:]/2
    c, d = q[..., :2]-q[..., 2:]/2, q[..., :2]+q[..., 2:]/2
    inter = (torch.minimum(b, d)-torch.maximum(a, c)).clamp_min(0).prod(-1)
    union = p[..., 2:].prod(-1)+q[..., 2:].prod(-1)-inter
    enclosing = (torch.maximum(b, d)-torch.minimum(a, c)).prod(-1)
    giou = inter/union-(enclosing-union)/enclosing
    return (l1_coef*((p-q).abs().sum(-1)*w).sum(-1)/w.sum()
        +giou_coef*((1-giou)*w).sum(-1)/w.sum())


def event_target_loss(central, targets, score, coeff, teacher_temperature,
                      student_temperature, weights):
    dist = event_geometry(central, targets, *coeff, weights)
    logp = (-dist/student_temperature).log_softmax(0)
    ranks = average_ranks(score.detach().cpu().numpy())
    logq = (-torch.as_tensor(ranks, device=central.device, dtype=central.dtype)
             /teacher_temperature).log_softmax(0).detach()
    loss = (logp.exp()*(logp-logq)).sum()
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite event rank KL')
    return loss, logp.exp(), logq.exp(), dist, logq, dict(
        reward_spread=float((score.max()-score.min()).detach()), strength=1.,
        s_ref=1., mode='rank', fixed_lambda=None, flat_noop=False,
        p_ref=logp.detach().exp().cpu(), q_rank=logq.detach().exp().cpu(),
        frozen_log_target=logq.cpu())


class SingleStep(OriginalSingleStep):
    def __init__(self, *args, support_mode='full', **kwargs):
        super().__init__(*args, **kwargs)
        assert support_mode in ['full', 'event'] and self.actuation == 'rkl'
        self.support_mode = support_mode

    def arrive(self, data, scheduled, temporal_provider=None, spatial_provider=None):
        if self.support_mode == 'full':
            return super().arrive(data, scheduled, temporal_provider, spatial_provider)
        actor = self.actor; before = actor.state()
        with torch.no_grad():
            if scheduled:
                ev, boxes, pre, layers, tc = central_with_candidates(actor, data)
            else:
                ev, boxes, pre = actor.values(data); layers = tc = None
        result = dict(prediction=detached(pre, 'cpu'), pre_state=detached(before, 'cpu'),
            pre_state_sha256=state_hash(before), updated=False, temporal_expert_read=False,
            spatial_expert_read=False, GT_read=False)
        out = pre
        if scheduled and self.fast:
            expert = temporal_provider(); assert expert['pixel_sha256'] == data['pixel_sha256']
            out, td = fast_rerank(pre, tc, expert)
            result.update(temporal=td, temporal_layers=layers, temporal_expert_read=True)
        # Current output is sealed before every spatial parameter write.
        result['output_prediction'] = detached(out, 'cpu'); post_ev = ev; post = pre
        if scheduled and self.slow:
            weights = event_weights(tc, data['frame_ids'])
            cs = []
            for k, delta in enumerate(self.deltas):
                _, _, cp = predict(self.model, data, {n:before[n]+delta[n] for n in before})
                if k == 0: assert torch.equal(cp['boxes'], pre['boxes'])
                cs.append(dict(prediction=detached(cp, 'cpu')))
            expert = spatial_provider(); assert expert['pixel_sha256'] == data['pixel_sha256']
            raw = rewards([c['prediction']['boxes'].numpy() for c in cs], expert['boxes'], expert['valid'])
            score = event_rewards([c['prediction']['boxes'].numpy() for c in cs],
                                  expert['boxes'], expert['valid'], weights)
            mass = float(weights[np.asarray(expert['valid'], bool)].sum())
            result.update(candidates=cs, rewards=score.tolist() if score is not None else None,
                valid_expert_frames=int(expert['valid'].sum()), spatial_expert_read=True,
                event_support=dict(candidates=tc, weights=weights.tolist(),
                    frame_ids=list(data['frame_ids']), candidate_count=len(tc),
                    weight_sum=float(weights.sum()), valid_reference_mass=mass,
                    positive_valid_frames=int((np.asarray(expert['valid'], bool)&(weights>0)).sum()),
                    unweighted_rewards=raw.tolist() if raw is not None else None,
                    no_update_reason=('empty_expert' if raw is None else 'zero_weighted_reference_mass')
                        if score is None else None, GT_read=False, detached=True))
            if score is not None:
                target = torch.stack([c['prediction']['boxes'] for c in cs]).cuda().detach()
                rt = torch.tensor(score, device='cuda', dtype=torch.float64).detach()
                evg, bg, pg = actor.values(data); assert torch.equal(bg.detach().cpu(), pre['boxes'])
                loss, pi, qi, dist, logtarget, meta = event_target_loss(bg, target, rt,
                    self.coeff, self.teacher_temperature, self.student_temperature, weights)
                grad = torch.autograd.grad(loss, [v for _, v in actor.named])
                if not all(torch.isfinite(g).all() for g in grad):
                    raise FloatingPointError('nonfinite event spatial gradient')
                scale, gn = update_scale(grad, 'rank', self.lr, 0.)
                selected = int(np.argmax(score)); flat = bool(float(np.ptp(score)) == 0.)
                u = dict(rank=average_ranks(score).tolist(), update_scale=scale,
                    global_gradient_norm=gn, loss_before=float(loss.detach()),
                    p=pi.detach().cpu(), q=qi.detach().cpu(), distances=dist.detach().cpu(),
                    gradients={n:g.detach().cpu() for (n,_),g in zip(actor.named,grad)},
                    gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grad))),
                    query_gradient_norm=float(grad[0].norm()),
                    LN_gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grad[1:]))),
                    coefficients=self.coeff, lr=self.lr, teacher_temperature=self.teacher_temperature,
                    student_temperature=self.student_temperature, candidate_targets_detached=True,
                    reward_detached=True, selected_index=selected, selected_flat_noop=flat,
                    selected_loss_before=float(dist[selected].detach()),
                    selected_gradients={n:torch.zeros_like(g).cpu() for (n,_),g in zip(actor.named,grad)},
                    srd_backward_calls=0, rkl_backward_calls=1, selected_target_detached=True,
                    objective_direction='rank_RKL', **meta)
                with torch.no_grad():
                    u.update(apply_direction(actor.named, grad, self.basis, average_ranks(score), scale, 'rkl'))
                    u.update(project_arrival(actor, self.arrival_center, None))
                    post_ev, bpost, post = actor.values(data)
                    dp = event_geometry(bpost, target, *self.coeff, weights)
                    lp = (-dp/self.student_temperature).log_softmax(0)
                    u.update(loss_after=float((lp.exp()*(lp-logtarget)).sum()),
                        selected_loss_after=float(dp[selected]),
                        full_distances_before=geometry(bg.detach(),target,*self.coeff).cpu(),
                        full_distances_after=geometry(bpost,target,*self.coeff).cpu())
                u.update(functional_movement(bg.detach(), bpost, target[selected]))
                result.update(update=u, updated=u['actuated'])
        after = actor.state()
        result.update(post_prediction=detached(post, 'cpu'), post_state=detached(after, 'cpu'),
            post_state_sha256=state_hash(after), parameter_displacement=float(torch.sqrt(sum(
                (after[n]-before[n]).square().sum() for n in before))),
            displacement_from_source=float(torch.sqrt(sum(
                (after[n]-actor.initial[n]).square().sum() for n in before))))
        assert torch.equal(result['output_prediction']['boxes'], result['prediction']['boxes'])
        return result, post_ev


class OnlineMethod(SingleStep):
    def __init__(self, model, deltas, *, steps=1, **kwargs):
        super().__init__(model, deltas, **kwargs)
        assert isinstance(steps, int) and 1 <= steps <= 10
        self.steps = steps

    def arrive(self, data, scheduled, temporal_provider=None, spatial_provider=None):
        self.arrival_center = self.actor.state(); cache = {}
        def spatial_once():
            if 'spatial' not in cache: cache['spatial'] = spatial_provider()
            return cache['spatial']
        result, ev = super().arrive(data, scheduled, temporal_provider, spatial_once)
        traces = []
        def record(x):
            return {k:x.get(k) for k in ['pre_state','post_state','pre_state_sha256',
                'post_state_sha256','updated','update','prediction','post_prediction',
                'candidates','rewards','valid_expert_frames','event_support']}
        if scheduled:
            traces.append(record(result))
            for step in range(1, self.steps):
                if traces[-1]['update'] is None: break
                saved_fast = self.fast; self.fast = False
                try: nxt, ev = super().arrive(data, True, None, spatial_once)
                finally: self.fast = saved_fast
                assert nxt['pre_state_sha256'] == traces[-1]['post_state_sha256']
                traces.append(record(nxt))
                for k in ['post_prediction','post_state','post_state_sha256','displacement_from_source']:
                    result[k] = nxt[k]
        result['update_steps'] = traces; result['updated'] = any(x['updated'] for x in traces)
        result['parameter_displacement'] = float(torch.sqrt(sum(
            (result['post_state'][n]-v).square().sum() for n,v in result['pre_state'].items())))
        count = sum(x['update'] is not None for x in traces)
        result['compute'] = dict(inner_steps=len(traces), spatial_candidate_replays=len(traces)*len(self.deltas),
            native_replays=1+max(0,len(traces)-1)+len(traces)*len(self.deltas)+2*count,
            backward_calls=count, rkl_backward_calls=count, srd_backward_calls=0,
            temporal_support_observations=len(traces) if self.support_mode=='event' else int(scheduled and self.fast),
            spatial_provider_calls=len(cache), temporal_provider_calls=int(scheduled and self.fast))
        return result, ev
