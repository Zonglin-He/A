"""Frozen proposal evidence, differentiable IoU energy, episodic C1 interface.

This is a proxy objective over DINO coordinates, not a native stochastic
policy, OPD implementation, or guarantee of expert/GT correctness.
"""
import math
import torch
from methods.decota_final_simplified_v1.objectives import xyxy
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.c1_enabling_tricks_v1 import TrickReplay

PROPOSAL_TEMPERATURE = 1.0
REWARD_TEMPERATURE = 1.0


def aligned_iou(pred, evidence):
    p, e = xyxy(pred), xyxy(evidence)
    intersection = (torch.minimum(p[..., 2:], e[..., 2:])
                    - torch.maximum(p[..., :2], e[..., :2])).clamp_min(0).prod(-1)
    pa = (p[..., 2:] - p[..., :2]).clamp_min(0).prod(-1)
    ea = (e[..., 2:] - e[..., :2]).clamp_min(0).prod(-1)
    return intersection / (pa + ea - intersection).clamp_min(1e-7)


class CriticEnergy:
    def __init__(self, expert, reference_boxes):
        self.frames = []
        self.frame_metadata = []
        for (_, position), observation in sorted(expert['observations'].items()):
            probe = observation['probe']
            evidence = torch.as_tensor(probe['boxes'], device=reference_boxes.device,
                                       dtype=reference_boxes.dtype).detach().clone()
            scores = torch.as_tensor(probe['target_scores'], device=reference_boxes.device,
                                     dtype=reference_boxes.dtype).detach().clone()
            assert evidence.shape == (len(scores), 4) and len(scores) <= 3
            assert torch.isfinite(evidence).all() and torch.isfinite(scores).all()
            assert 0 <= position < len(reference_boxes)
            valid = (evidence[:, 2:] > 0).all(1)
            evidence, scores = evidence[valid], scores[valid]
            if not len(scores):
                continue
            assert ((scores >= 0) & (scores <= 1)).all()
            log_weights = torch.log_softmax(scores / PROPOSAL_TEMPERATURE, 0)
            self.frames.append((position, evidence, log_weights))
            w = log_weights.exp()
            self.frame_metadata.append(dict(position=position, proposals=len(scores),
                old_admitted=bool(probe['accepted']), context_active=bool(observation['receipt']['context_active']),
                weights=w.cpu().tolist(), scores=scores.cpu().tolist(),
                evidence_entropy=float(-(w * log_weights).sum())))
        self.empty = not self.frames

    def __call__(self, boxes):
        if self.empty:
            return boxes.sum() * 0
        energies = [-torch.logsumexp(logw + aligned_iou(boxes[position], evidence)
                                    / REWARD_TEMPERATURE, 0)
                    for position, evidence, logw in self.frames]
        return torch.stack(energies).mean()

    @torch.no_grad()
    def diagnostics(self, boxes):
        values = []
        for position, evidence, logw in self.frames:
            overlap = aligned_iou(boxes[position], evidence)
            energy = -torch.logsumexp(logw + overlap / REWARD_TEMPERATURE, 0)
            values.append(dict(position=position, overlaps=overlap.cpu().tolist(),
                max_IoU=float(overlap.max()), weighted_IoU=float((logw.exp() * overlap).sum()),
                energy=float(energy), all_zero_overlap=bool((overlap == 0).all())))
        return values


@torch.enable_grad()
def fit_critic(base, initial, expert, ids, key):
    base.restore(initial)
    replay = TrickReplay(base, ids, key, {})
    origin = replay.state()
    params = [p for _, p in replay.named]
    assert sum(p.numel() for p in params) == 1792
    objective = CriticEnergy(expert, base.zero['boxes'])
    optimizer = torch.optim.Adam(params, lr=.03, betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)
    chosen, best, selected = detached(origin), math.inf, 0
    path, backwards = [], 0
    try:
        for step in range(1 if objective.empty else 11):
            optimizer.zero_grad(set_to_none=True)
            values = replay.values()
            loss = objective(values['boxes'])
            lv = float(loss.detach())
            assert math.isfinite(lv), 'Nonfinite critic energy'
            if lv < best:
                best, selected, chosen = lv, step, replay.state()
            entry = dict(step=step, loss=lv, state=detached(replay.state(), 'cpu'),
                boxes=detached(values['boxes'], 'cpu'),
                evidence_diagnostics=objective.diagnostics(values['boxes']))
            path.append(entry)
            if objective.empty or step == 10:
                break
            grads = torch.autograd.grad(loss, params)
            gradient = torch.cat([g.reshape(-1) for g in grads]).detach()
            assert torch.isfinite(gradient).all()
            for p, g in zip(params, grads):
                p.grad = g.detach()
            before = torch.cat([p.reshape(-1) for p in params]).detach().clone()
            optimizer.step()
            after = torch.cat([p.reshape(-1) for p in params]).detach()
            entry['update'] = dict(gradient=gradient.cpu(), raw=(after - before).cpu(),
                adam_steps={n: int(optimizer.state[p]['step']) for n, p in replay.named})
            assert all(torch.isfinite(p).all() for p in params)
            backwards += 1
        return dict(arm='critic', initial=detached(origin, 'cpu'), state=detached(chosen, 'cpu'),
            selected_step=selected, final=path[selected]['boxes'], path=path,
            skipped=objective.empty, gradient_calls=backwards, parameter_count=1792,
            frame_metadata=objective.frame_metadata, GT_used=False)
    finally:
        replay.restore(origin)
