"""Frozen negative-evidence targets for a query-local spatial comparison.

No model, optimizer, data loader, GT, or state writer lives here. Boxes use the
existing normalized cxcywh convention. Candidate zero is the current center.
The geometric softmax is a compatibility surrogate, not a native tube policy.
Torch is imported only by the differentiable functions.

Scientific lock: lambda = 1 in raw IoU-difference units. Both support modes use
the same per-observation positive part. Global averages this evidence across
valid observations and uses full-clip geometry; local uses one distribution per
valid observation. Targets and anchors are frozen within each inner step, then
rebuilt from refreshed candidates at the next inner step.
"""

import numpy as np

LAMBDA = 1.0
BLOCKS = ("query", "norm1", "norm3", "norm4")


def negative_evidence(candidates, expert_boxes, valid):
    """Return IoU[j,k] and [IoU[j,0]-IoU[j,k]]_+ on valid frames.

    candidates: [K,F,4]; expert_boxes: [F,4]; valid: [F]. Invalid expert
    positions are never inspected for finiteness or used as negative evidence.
    Returned frame arrays are [J,K], where J is the valid-frame count. No
    confidence, temporal annotation, or expert-absence signal is manufactured.
    """
    c = np.asarray(candidates, dtype=np.float64)
    e = np.asarray(expert_boxes, dtype=np.float64)
    v = np.asarray(valid)
    if c.ndim != 3 or c.shape[-1] != 4 or not c.shape[0] or not c.shape[1]:
        raise ValueError("candidates must have nonempty shape [K,F,4]")
    if e.shape != c.shape[1:] or v.shape != (c.shape[1],) or v.dtype != np.bool_:
        raise ValueError("expert boxes / boolean validity must align with frames")
    if not np.isfinite(c).all() or (c[..., 2:] <= 0).any():
        raise ValueError("candidate boxes must be finite with positive sizes")
    positions = np.flatnonzero(v)
    if not len(positions):
        empty = np.empty((0, c.shape[0]), dtype=np.float64)
        return dict(framewise_rewards=empty, e_jk=empty.copy(),
                    valid_positions=positions, global_e=np.zeros(c.shape[0]))
    observed = e[positions]
    if not np.isfinite(observed).all() or (observed[:, 2:] <= 0).any():
        raise ValueError("valid expert boxes must be finite with positive sizes")
    q = c[:, positions]
    lo = np.maximum(q[..., :2] - q[..., 2:] / 2,
                    observed[None, ..., :2] - observed[None, ..., 2:] / 2)
    hi = np.minimum(q[..., :2] + q[..., 2:] / 2,
                    observed[None, ..., :2] + observed[None, ..., 2:] / 2)
    inter = np.maximum(hi - lo, 0).prod(-1)
    union = q[..., 2:].prod(-1) + observed[None, ..., 2:].prod(-1) - inter
    reward = (inter / np.maximum(union, 1e-12)).T
    evidence = np.maximum(reward[:, :1] - reward, 0)
    if not np.isfinite(reward).all() or (evidence > 1 + 1e-12).any():
        raise FloatingPointError("invalid IoU negative evidence")
    return dict(framewise_rewards=reward, e_jk=evidence,
                valid_positions=positions, global_e=evidence.mean(0))


def geometry_distances(central, candidates, coeff, *, mode, valid_positions=()):
    """Existing L1+GIoU arithmetic, full clip [K] or valid frames [J,K].

    Candidate boxes are detached. Global arithmetic retains the old order of
    reductions so it can be checked against the sealed geometry implementation.
    Only local support changes: it omits the full-clip frame mean.
    """
    import torch

    if mode not in ("global", "local"):
        raise ValueError("mode must be global or local")
    if central.ndim != 2 or central.shape[-1] != 4:
        raise ValueError("central must have shape [F,4]")
    q = torch.as_tensor(candidates, device=central.device, dtype=central.dtype).detach()
    if q.ndim != 3 or q.shape[1:] != central.shape or q.shape[0] == 0:
        raise ValueError("candidates must have shape [K,F,4]")
    coefficients = tuple(float(x) for x in coeff)
    if len(coefficients) != 2 or not all(np.isfinite(x) and x >= 0 for x in coefficients):
        raise ValueError("coeff must contain nonnegative finite L1 and GIoU weights")
    if not torch.isfinite(central).all() or not torch.isfinite(q).all():
        raise FloatingPointError("nonfinite spatial geometry")
    if (central[..., 2:] <= 0).any() or (q[..., 2:] <= 0).any():
        raise ValueError("geometry requires positive box sizes")
    p = central.unsqueeze(0)
    a, b = p[..., :2] - p[..., 2:] / 2, p[..., :2] + p[..., 2:] / 2
    c, d = q[..., :2] - q[..., 2:] / 2, q[..., :2] + q[..., 2:] / 2
    inter = (torch.minimum(b, d) - torch.maximum(a, c)).clamp_min(0).prod(-1)
    union = p[..., 2:].prod(-1) + q[..., 2:].prod(-1) - inter
    enclosing = (torch.maximum(b, d) - torch.minimum(a, c)).prod(-1)
    giou = inter / union - (enclosing - union) / enclosing
    l1, g = (p - q).abs().sum(-1), 1 - giou
    if mode == "global":
        distance = coefficients[0] * l1.mean(-1) + coefficients[1] * g.mean(-1)
    else:
        positions = torch.as_tensor(valid_positions, device=central.device, dtype=torch.long)
        if positions.ndim != 1 or ((positions < 0) | (positions >= len(central))).any():
            raise ValueError("valid positions out of range")
        if len(positions) > 1 and not (positions[1:] > positions[:-1]).all():
            raise ValueError("valid positions must be unique and increasing")
        distance = (coefficients[0] * l1 + coefficients[1] * g)[:, positions].T
    if not torch.isfinite(distance).all():
        raise FloatingPointError("nonfinite spatial distance")
    return distance


def make_anchor(central, candidates, expert_boxes, valid, coeff, *, mode,
                student_temperature=1.0):
    """Freeze step-local candidates, p0, logq and observed-frame evidence.

    mode='global': e[k] = mean_j e[j,k], full-clip geometric p0[k].
    mode='local': e[j,k] and same-observation geometric p0[j,k].
    Caller rebuilds this object at every inner step, never after its update.
    """
    import torch

    temperature = float(student_temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("student temperature must be finite and positive")
    target = torch.as_tensor(candidates, device=central.device, dtype=central.dtype).detach().clone()
    expert = expert_boxes.detach().cpu().numpy() if torch.is_tensor(expert_boxes) else expert_boxes
    validity = valid.detach().cpu().numpy() if torch.is_tensor(valid) else valid
    evidence = negative_evidence(target.cpu().numpy(), expert, validity)
    positions = evidence["valid_positions"]
    distance = geometry_distances(central.detach(), target, coeff, mode=mode,
                                  valid_positions=positions).detach()
    logp0 = (-distance / temperature).log_softmax(-1).detach()
    raw_e = evidence["global_e"] if mode == "global" else evidence["e_jk"]
    e = torch.as_tensor(raw_e, device=central.device, dtype=central.dtype).detach().clone()
    # Clone in the no-evidence case: no exp/log round trip or artificial entropy target.
    logq = logp0.clone() if not (e != 0).any() else (logp0 - LAMBDA * e).log_softmax(-1).detach()
    return dict(mode=mode, candidates=target, coeff=tuple(float(x) for x in coeff),
                student_temperature=temperature, lambda_value=LAMBDA,
                logp0=logp0, logq=logq, e=e,
                e_jk=torch.as_tensor(evidence["e_jk"], device=central.device,
                                     dtype=central.dtype).detach().clone(),
                framewise_rewards=torch.as_tensor(evidence["framewise_rewards"],
                                     device=central.device, dtype=central.dtype).detach().clone(),
                valid_positions=torch.as_tensor(positions, device=central.device,
                                     dtype=torch.long).detach().clone(),
                center=central.detach().clone(), empty_evidence=len(positions) == 0)


def negative_loss(central, anchor):
    """Evaluate KL(p||frozen q); use the SAME anchor for loss_after.

    The gradient equals that of KL(p||p0) + lambda*E_p[e]. The former differs
    only by the step-constant log normalizer; both values are returned for audit.
    No-valid-frame evidence yields an explicit connected zero loss, allowing the
    caller to skip the update. Zero evidence at the anchor also returns an exact
    connected zero; after a genuine movement, the frozen-anchor KL is retained.
    """
    import torch

    if anchor["lambda_value"] != LAMBDA:
        raise ValueError("lambda is fixed at 1")
    if central.device != anchor["candidates"].device or central.dtype != anchor["candidates"].dtype:
        raise ValueError("anchor device and dtype must match central")
    distance = geometry_distances(central, anchor["candidates"], anchor["coeff"],
                    mode=anchor["mode"], valid_positions=anchor["valid_positions"])
    logp = (-distance / anchor["student_temperature"]).log_softmax(-1)
    logp0, logq = anchor["logp0"].detach(), anchor["logq"].detach()
    e = anchor["e"].detach()
    zero_initial = not (e != 0).any() and torch.equal(central.detach(), anchor["center"])
    if anchor["empty_evidence"] or zero_initial:
        loss = central.sum() * 0
        proximal_plus_penalty = loss
    else:
        loss = (logp.exp() * (logp - logq)).sum(-1).mean()
        proximal_plus_penalty = (logp.exp() * (logp - logp0 + LAMBDA * e)).sum(-1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite negative-evidence loss")
    return dict(loss=loss, logp=logp, logp0=logp0, logq=logq, distance=distance,
                e=e, e_jk=anchor["e_jk"], framewise_rewards=anchor["framewise_rewards"],
                valid_positions=anchor["valid_positions"], mode=anchor["mode"],
                lambda_value=LAMBDA, empty_evidence=anchor["empty_evidence"],
                exact_initial_noop=bool(zero_initial),
                proximal_plus_penalty=proximal_plus_penalty)


def parameter_blocks(state):
    """Validate the existing 1792-state layout and return the four key groups."""
    groups = {"query": ("spatial.query_residual",)}
    for name in BLOCKS[1:]:
        groups[name] = tuple(f"spatial.layers.5.{name}.{part}" for part in ("weight", "bias"))
    if set(state) != {key for keys in groups.values() for key in keys}:
        raise ValueError("state must contain exactly the seven existing spatial parameters")
    for key, value in state.items():
        if tuple(value.shape) != (256,):
            raise ValueError(f"unexpected parameter shape: {key}")
    return groups


def block_only_state(pre_state, post_state, block):
    """Independent pre-state clone with one final block replaced by post-state.

    This is a finite update counterfactual; its four output effects need not add
    to the full update and are not Jacobian contributions. It never writes into
    a model, either input mapping, or a future query's persistent state.
    """
    groups = parameter_blocks(pre_state)
    parameter_blocks(post_state)
    if block not in groups:
        raise ValueError(f"unknown spatial block: {block}")
    def clone(value):
        return value.detach().clone() if hasattr(value, "detach") else np.array(value, copy=True)
    return {key: clone(post_state[key] if key in groups[block] else value)
            for key, value in pre_state.items()}
