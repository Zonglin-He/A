"""Compact, real Adam trajectories with the fixed best/last state semantics."""

import math
import torch

from .objectives import SpatialLoss, project, temporal_loss
from .tensors import detached


@torch.enable_grad()
def _fit(replay, zero, loss_fn, *, lr, steps, temporal, empty=False, trace=False):
    replay.restore(replay.initial)
    params = [p for _, p in replay.named]
    cls = torch.optim.AdamW if temporal else torch.optim.Adam
    opt = cls(params, lr=lr, betas=(.9, .999), eps=1e-4 if temporal else 1e-8,
              weight_decay=0.)
    losses, history, best_loss, selected_step = [], [], math.inf, 0
    chosen, last, output = replay.state(), replay.state(), zero
    failure, backwards, evaluations = None, 0, 0
    try:
        for step in range((0 if empty else steps) + 1):
            opt.zero_grad(set_to_none=True)
            value = replay.values()
            evaluations += 1
            loss = loss_fn(value)
            lv = float(loss.detach())
            if not math.isfinite(lv):
                failure = "nonfinite"
                break
            losses.append(lv)
            # Temporal selects actual last state; spatial includes step zero in best.
            if temporal or lv < best_loss:
                chosen, selected_step = replay.state(), step
                output = detached({"boxes": value["boxes"], "logits": value["logits"]})
                best_loss = lv
            if trace:
                history.append(dict(step=step, loss=lv, state=detached(replay.state(), "cpu"),
                                    boxes=detached(value["boxes"], "cpu"),
                                    logits=detached(value["logits"], "cpu")))
            if empty or step == steps:
                last = replay.state()
                break
            loss.backward()
            backwards += 1
            grads = [p.grad for p in params if p.grad is not None]
            if temporal and len(grads) != len(params):
                raise RuntimeError("A trainable temporal-head parameter is disconnected")
            if grads and not torch.stack([torch.isfinite(g).all() for g in grads]).all():
                failure = "nonfinite_gradient"
                break
            opt.step()
        if not losses:
            raise RuntimeError("Invalid step-zero loss; not a successful no-op")
        if failure:
            chosen, selected_step, output = replay.initial, 0, zero
        changed = any(not torch.equal(chosen[n], v) for n, v in replay.initial.items())
        return dict(state=detached(chosen), initial_state=detached(replay.initial),
                    last_state=detached(last), selected_step=selected_step,
                    losses=losses, final=output, path=history, failure=failure,
                    skipped=empty, backwards=backwards, evaluations=evaluations,
                    parameter_changed=changed, source_restored=True, GT_online=False)
    finally:
        replay.restore(replay.initial)


def fit_spatial(replay, zero, anchors, config, trace=False):
    loss = SpatialLoss(anchors, zero["boxes"])
    return _fit(replay, zero, lambda value: loss(value["boxes"]),
                lr=config.spatial_lr, steps=config.spatial_steps, temporal=False,
                empty=loss.empty, trace=trace)


def fit_temporal(replay, records, config, trace=False):
    evidence = project(replay.zero["logits"], replay.zero["actions"], records, config)
    result = _fit(replay, replay.zero,
                  lambda value: temporal_loss(value["logits"], evidence, config.margin),
                  lr=config.temporal_lr, steps=config.temporal_steps,
                  temporal=True, trace=trace)
    result["evidence"] = evidence
    state = {n: v + config.eta * (result["state"][n] - v)
             for n, v in replay.initial.items()}
    try:
        replay.restore(state)
        with torch.no_grad():
            final = detached(replay.values())
    finally:
        replay.restore(replay.initial)
    result.update(shrunk_state=state, shrunk=final, eta=config.eta,
                  shrunk_parameter_changed=any(not torch.equal(state[n], v)
                                               for n, v in replay.initial.items()))
    return result
